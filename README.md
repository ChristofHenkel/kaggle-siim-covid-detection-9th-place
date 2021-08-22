# kaggle-siim-covid19-detection

## Summary of 9th place solution (Team Watercooled)

This repository summarizes our solution which consists of a blend of classification models for and object detection models
In order to reproduce our solution its sufficient to run the following steps.
Please also find our describing summary on kaggle (https://www.kaggle.com/c/siim-covid19-detection/discussion/265583)


### Preprocessing
- read dicom and resize to 0.25x
- read dicom and resize to 0.5x
Those already preprocessed datasets can be downloaded from https://www.kaggle.com/christofhenkel/siim-covid19-detection-025-downscaled and https://www.kaggle.com/christofhenkel/siim-covid19-detection-05-downscaled

Move to subfolders input/train_05, input/train_25, input/test_25, input/test_05 or adjust paths in the config files.

### Study models

```sh
train5fold.sh -C cfg_ch_study_xcit_2e15
train5fold.sh -C cfg_ch_study_xcit_2d
train5fold.sh -C cfg_ps_study_ben_7
train5fold.sh -C cfg_ps_study_7
train5fold.sh -C cfg_ch_study_12d
train5fold.sh -C cfg_ch_study_12d
train5fold.sh -C cfg_ch_study_7_1024b5
train5fold.sh -C cfg_ch_study_7_1024b7_2
```


### Image models

#### EfficientDet
```sh
train5fold.sh -C cfg_ps_35
train5fold.sh -C cfg_ps_27
train5fold.sh -C cfg_ps_13
train5fold.sh -C cfg_ps_9
```

#### Yolov5

```sh
yolov5/run_ch_1b.sh
yolov5/run_ch_2b.sh
yolov5/run_ch_3b.sh
yolov5/run_ch_7_2cls_2.sh
```
