# Credit to https://blog.roboflow.com/how-to-train-detectron2/ 
import os
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from models import trainer_model

def main(output_folder_path, train_img_folder, train_annotation_json, val_img_folder, val_annotation_json):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    # Prepare datasets
    register_coco_instances("train", {}, train_annotation_json, train_img_folder)
    register_coco_instances("val", {}, val_annotation_json, val_img_folder)
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)

    # Prepare output folder
    cfg.OUTPUT_DIR = output_folder_path
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Model parameters
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025 
    
    cfg.SOLVER.MAX_ITER = 1000   

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

    cfg.TEST.EVAL_PERIOD = 20 
    cfg.TEST.VISUALIZATION_PERIOD = 100
    cfg.TEST.VISUALIZATION_NB_SHOW = 2 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

    # Train
    trainer = trainer_model.Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    output_folder_path = "output"
    train_img_folder = "data/train_coco"
    train_annotation_json = "data/train_coco/annotations.json"
    val_img_folder = "data/val_coco"
    val_annotation_json = "data/val_coco/annotations.json"
    main(output_folder_path, train_img_folder, train_annotation_json, val_img_folder, val_annotation_json)