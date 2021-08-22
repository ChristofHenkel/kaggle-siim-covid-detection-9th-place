from types import SimpleNamespace
import numpy as np

cfg = SimpleNamespace(**{})

# dataset
cfg.classes = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]
cfg.labels_img = np.array(["negative", "typical", "indeterminate", "atypical"])
cfg.dataset = "base_ds"
cfg.suffix = ".png"
cfg.n_classes = 4
cfg.batch_size = 32
cfg.val_df = None
cfg.test = False
cfg.test_df = None
cfg.batch_size_val = None
cfg.normalization = "none"
cfg.train_aug = False
cfg.val_aug = False
cfg.test_augs = False
cfg.img_aug = False
cfg.scale = 0.25
cfg.data_sample = -1
cfg.box_min_conf = 0.01

# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.pretrained_weights_strict = True
cfg.pool = "avg"
cfg.train = True
cfg.val = True
cfg.in_chans = 6
cfg.epoch_weights = None
cfg.calc_loss = True
cfg.gem_p_trainable = False
cfg.study_weight = 1
cfg.backbone_kwargs = {}

cfg.pretrained_convhead = False

cfg.alpha = 1
cfg.cls_loss_pos_weight = None
cfg.train_val = True
cfg.eval_epochs = 1
cfg.eval_train_epochs = 5
cfg.drop_path_rate = 0.0
cfg.drop_rate = 0.0
cfg.dropout = 0.0
cfg.attn_drop_rate = 0.0
cfg.warmup = 0
cfg.label_smoothing = 0
cfg.pp_transformation = "softmax"

# training
cfg.fold = 0
cfg.val_fold = -1
cfg.lr = 1e-4
cfg.weight_decay = 0
cfg.optimizer = "Adam"  # "Adam", "AdamW"
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False
cfg.simple_eval = False
cfg.do_test = True
cfg.do_seg = False
cfg.eval_ddp = True
cfg.clip_grad = 0
cfg.debug = False
cfg.save_val_data = True
cfg.reload_train_loader = False
# ressources
cfg.find_unused_parameters = False
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.dp = False
cfg.num_workers = 4
cfg.drop_last = True
cfg.pin_memory = False

# logging,
cfg.calc_metric = True
cfg.sgd_nesterov = True
cfg.sgd_momentum = 0.9
cfg.clip_mode = "norm"
cfg.data_sample = -1

cfg.channel_shuffle = 0
cfg.on_off_merge = False

cfg.mixup = 0
cfg.cutmix = 0
cfg.mosaic = 0
cfg.boxmix = 0

cfg.mix_beta = 0.5
cfg.mixadd = False

cfg.mix_same = False

cfg.cutmix_data = 0
cfg.mixup_data = 0
cfg.mixup_model = 0
cfg.mixup_model_2 = 0
cfg.channel_mixup = 0
cfg.mixup_beta = 1
cfg.halfmix = 0

cfg.num_scales = 3
cfg.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
cfg.box_loss_weight = 50

cfg.create_submission = True

cfg.on_only = False

cfg.stride = None

cfg.loss = "bce"

cfg.class_weights = None

cfg.tta = []

basic_cfg = cfg
