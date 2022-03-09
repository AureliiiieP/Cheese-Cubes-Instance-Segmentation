# Cheese-Cubes-Instance-Segmentation
Small project on custom small dataset. 
We use instance segmentation (detectron2) to detect and classify flavors of French delicious cheese cubes. 

Disclaimer. This is a tiny project made in my free time when I was trying to learn instance segmentation. I'm in the process of cleaning, adding documentation to this repo. Please be patient with me !

## Dataset
A very small toy dataset was created by annotating 39 photos of cheese cubes of 5 different flavors :
- Olive
- Jambon Cru (Cured Ham)
- Bleu (Blue cheese)
- Camembert
- Cheddar

The data was annotated using [labelme](https://github.com/wkentaro/labelme).

Please contact me if you want to download the dataset !

Example of image :
<img src="doc/cheese-thrown-in-box.PNG" width="500" />

## Convert the labels 

The labels need to be converted to COCO format before training. Please run :
```
python3 utils/labelme2coco.py "data/train" "data/train_coco" data/labels.txt
```

## Training
Please set data paths, output folder path inside the file and run :
```
python3 train.py
```

Note. Training can be quite long even with small dataset. Metrics can be stuck at 0 / Nan for long time.
Please try changing parameters in train.py to save up time and accelerate convergence.

## Inference

## Example of results
<img src="doc/result_example.png" width="500" />

## Folder organization
