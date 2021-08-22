from default_config import basic_cfg
import albumentations as A
import os

cfg = basic_cfg
cfg.debug = True

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.data_dir = "input/"
cfg.data_folder = cfg.data_dir + "train_05/"
cfg.train_df = cfg.data_dir + "train_image_level_folded_v3.csv"
cfg.output_dir = f"output/models/{os.path.basename(__file__).split('.')[0]}"

cfg.test = True
cfg.test_df = cfg.data_dir + "test_image_level_v2.csv"
cfg.test_data_folder = cfg.data_dir + "test_05/"
cfg.create_submission = False

# OPTIMIZATION & SCHEDULE
cfg.lr = 0.0003
cfg.optimizer = "AdamW"
cfg.epochs = 10
cfg.batch_size = 4
cfg.mixed_precision = False
cfg.pin_memory = False
# MODEL
cfg.model = "ps_study_1"
cfg.backbone = "tf_efficientnet_b7_ns"
cfg.in_channels = 1
# DATASET
cfg.dataset = "ch_ds_study_2"
cfg.normalization = "image"
cfg.eval_train_epochs = 5
cfg.scale = 0.5

cfg.pool = "gem"
cfg.gem_p_trainable = True

# TRAINING
cfg.gpu = 0
cfg.num_workers = 16

cfg.mixup = 0
cfg.image_size = (1024, 1024)

cfg.seg_dim = 1
cfg.seg_weight = 50

cfg.pp_transformation = "softmax"

cfg.train = True

cfg.train_aug = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),
        A.Resize(1024, 1024, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5
        ),
        A.Cutout(
            num_holes=8,
            max_h_size=int(cfg.image_size[0] * 0.1),
            max_w_size=int(cfg.image_size[1] * 0.1),
            p=1,
        ),
        A.RandomCrop(768, 768),
    ],
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0),
)

cfg.val_aug = A.Compose(
    [A.Resize(cfg.image_size[0], cfg.image_size[1], p=1.0),],
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0),
)
