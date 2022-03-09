import os, cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.engine.hooks import HookBase
from detectron2.data import DatasetMapper, build_detection_test_loader, build_detection_train_loader, MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer
from .data_aug import build_train_aug

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._best_val_loss = float('inf')
    
    def _do_loss_eval(self):
        total = len(self._data_loader)
        losses = []
        for idx, inputs in enumerate(self._data_loader): 
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        return losses
            
    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        losses = [loss for loss in metrics_dict.values()]
        total_losses_reduced = sum(losses)
        return total_losses_reduced
    
    def _plot_loss(self):
        if self.trainer.iter > self.trainer.cfg.TEST.EVAL_PERIOD and self.trainer.iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            train_losses = [elem[0] for elem in self.trainer.storage.history("total_loss")._data]
            train_loss = [np.mean(train_losses[i*self.trainer.cfg.TEST.EVAL_PERIOD:(i+1)*self.trainer.cfg.TEST.EVAL_PERIOD]) for i in range(len(train_losses)//self.trainer.cfg.TEST.EVAL_PERIOD)]
            val_loss = [elem[0] for elem in self.trainer.storage.history("validation_loss")._data]
            plt.plot(train_loss)
            plt.plot(val_loss)
            plt.xlabel("Evaluation ticks")
            plt.ylabel("Loss")
            plt.show()
            plt.savefig(os.path.join(self.trainer.cfg.OUTPUT_DIR, "val_loss.png"))
            plt.close()
        
    def _pred_img(self):
        # Save prediction on some images of the validation set at regular intervals
        dataset_dicts = DatasetCatalog.get(self.trainer.cfg.DATASETS.TEST[0])
        metadata = MetadataCatalog.get(self.trainer.cfg.DATASETS.TEST[0])
        predictor = DefaultPredictor(self.trainer.cfg)
        for i,d in enumerate(dataset_dicts):
            if i < self.trainer.cfg.TEST.VISUALIZATION_NB_SHOW :
                im = cv2.imread(d["file_name"])
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1],
                            scale=0.5, 
                            instance_mode=ColorMode.IMAGE_BW,
                            metadata = metadata
                )
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                img = out.get_image()[:, :, ::-1]
                img = img.copy()
                count = len(outputs['instances'])
                h,w,c = img.shape
                cv2.imwrite(os.path.join(self.trainer.cfg.OUTPUT_DIR, str(self.trainer.iter) + "_" + os.path.basename(d["file_name"])),img)
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        self._plot_loss()
        # Save if best model based on lowest val loss
        if self.trainer.iter > self.trainer.cfg.TEST.EVAL_PERIOD :
            val_loss = self.trainer.storage.history("validation_loss")._data[-1][0]
            if val_loss  < self._best_val_loss :
                torch.save(self.trainer.model.state_dict(), os.path.join(self.trainer.cfg.OUTPUT_DIR, 'best.pth'))
                self._best_val_loss = val_loss
        # Save prediction on some images of validation
        if self.trainer.iter >= self.trainer.cfg.TEST.VISUALIZATION_PERIOD and self.trainer.iter % self.trainer.cfg.TEST.VISUALIZATION_PERIOD == 0 :
            self._pred_img()
        

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
    
    def train(self):
        super().train()
