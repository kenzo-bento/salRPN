## Instructions for dataset download and saliency generation
The following instructions are to help set up the proper datasets file structure and downloads. The files coco.sh installs the MS COCO dataset, and the PASCAL VOC dataset can be found on Kaggle. runProtoObject.m generates saliency maps for the respective datasets.

## Finalized file structure:
```
datasets/
├── coco/
|   ├── annotations/
|   ├── images/
|   └── sal/
|       └── ProtoObject/
|           ├── train_maps/
|           └── val_maps/
└── voc/
    ├── VOC2012_test/
    |   ├── Annotations/
    |   ├── ImageSets/
    |   ├── JPEGImages/
    |   └── sal/
    |       └── ProtoObject/
    |            └── val_maps/
    └── VOC2012_train_val/ (Matches structure of VOC2012_test/)
```

## Downloading MS COCO dataset:
Navigate to the datasets directory, grant coco.sh execution permissions, and run the program.
```bash
cd ./datasets
chmod +x coco.sh
./coco.sh
```

## Downloading PASCAL VOC dataset:
The PASCAL VOC 2012 dataset can be found and downloaded via Kaggle here: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset

## Saliency generation:
Download the repository from https://github.com/csmslab/proto-object-saliency-texture (Uejima et. al, 2020). Run runProtoObject.m after specifying the path towards the downloaded saliency generation folder on line 7. Specify input and output folders that are noted in the runProtoObject.m file, in line 14 and 16 respectively, before running the matlab file on each dataset.


