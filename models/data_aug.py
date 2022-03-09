import detectron2.data.transforms as T

def build_train_aug(cfg):
    # Data Augmentation
    augs = []
    augs.append(T.RandomBrightness(0.9, 1.1))
    augs.append(T.RandomFlip(prob=0.5, horizontal = True))
    augs.append(T.RandomFlip(prob=0.5, horizontal = False, vertical = True))
    augs.append(T.RandomRotation([0,360]))
    augs.append(T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING))
    return augs