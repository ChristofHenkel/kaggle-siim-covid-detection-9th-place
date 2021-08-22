from augmentations import CustomCutout
from default_config import basic_cfg
import albumentations as A
import os

cfg = basic_cfg
cfg.debug = True

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = "input/"
cfg.data_folder = cfg.data_dir + "train_025/"
cfg.train_df = cfg.data_dir + "train_image_level_folded_v3.csv"
cfg.output_dir = f"output/models/{os.path.basename(__file__).split('.')[0]}"

cfg.test = True
cfg.test_df = cfg.data_dir + "test_image_level_v2.csv"
cfg.test_data_folder = cfg.data_dir + "test_025/"

# OPTIMIZATION & SCHEDULE
cfg.lr = 0.002
cfg.optimizer = "AdamW"
cfg.epochs = 14
cfg.batch_size = 32
cfg.mixed_precision = True
cfg.pin_memory = False
# MODEL
cfg.model = "ps_mdl_3"
cfg.backbone = "tf_efficientdet_d3"
cfg.in_channels = 1
# DATASET
cfg.dataset = "ps_ds_1"
cfg.normalization = "simple"

cfg.scale = 0.25

cfg.pool = "gem"
cfg.gem_p_trainable = True

# TRAINING
cfg.gpu = 0
cfg.num_workers = 16

cfg.image_size = (512, 512)

cfg.study_weight = 0.1

cfg.mixup = 0.0
cfg.cutmix = 0.0
cfg.mosaic = 0.0
cfg.mixadd = False

cfg.loss = "bce"
cfg.pp_transformation = "softmax"

cfg.train = True

cfg.train_aug = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),
        A.Resize(cfg.image_size[0], cfg.image_size[1], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5
        ),
        CustomCutout(
            fill_value=0,
            bbox_removal_threshold=0.25,
            min_cutout_size=64,
            max_cutout_size=256,
            p=1.0,
        )
    ],
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0),
)

cfg.val_aug = A.Compose(
    [A.Resize(cfg.image_size[0], cfg.image_size[1], p=1.0),],
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0),
)
