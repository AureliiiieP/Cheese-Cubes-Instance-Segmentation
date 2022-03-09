import detectron2
import numpy as np
import os, json, cv2
import glob
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

import argparse

def main(args):
    # Create test dataset
    register_coco_instances("test", {}, args.annotation, args.input_dir)
    dataset_dicts = DatasetCatalog.get("test")
    metadata = MetadataCatalog.get("test")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
    cfg.MODEL.WEIGHTS = args.model_weight
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    os.makedirs(args.output_dir, exist_ok = True)

    for d in dataset_dicts:
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
        cv2.putText(img, str(count), (w-200, 150), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(img, str(count*17) + " kcal", (w-400, 250), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(d["file_name"])),img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-input_dir", help="input annotated directory")
    parser.add_argument("-annotation", help="annotation")
    parser.add_argument("-model_weight", help="model weight to load")
    parser.add_argument("-output_dir", help="output dataset directory")
    args = parser.parse_args()
    main(args)