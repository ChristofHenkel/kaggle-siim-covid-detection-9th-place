import torch
from torch import nn
import timm
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from segmentation_models_pytorch.unet.model import UnetDecoder, SegmentationHead


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            in_chans=1,
            drop_path_rate=cfg.drop_path_rate,
        )

        self.cfg = cfg

        decoder_channels = (256, 128, 64, 32, 16)

        if "efficientnet_b0" in cfg.backbone:
            encoder_channels = (1, 16, 24, 40, 112, 320)
            backbone_out = 320
        elif "efficientnet_b3" in cfg.backbone:
            encoder_channels = (1, 24, 32, 48, 136, 384)
            backbone_out = 384
        elif "efficientnet_b5" in cfg.backbone:
            encoder_channels = (1, 24, 40, 64, 176, 512)
            backbone_out = 512
        elif "efficientnet_b7" in cfg.backbone:
            encoder_channels = (1, 32, 48, 80, 224, 640)
            backbone_out = 640

        elif "efficientnet_b7" in cfg.backbone:
            encoder_channels = (1, 32, 48, 80, 224, 640)
            backbone_out = 640
        elif "tf_efficientnetv2_m" in cfg.backbone:
            encoder_channels = (1, 24, 48, 80, 176, 512)
        elif "tf_efficientnetv2_l" in cfg.backbone:
            encoder_channels = (1, 32, 64, 96, 224, 640)
            backbone_out = 640
        elif "efficientnet_b8" in cfg.backbone:
            backbone_out = 704
            encoder_channels = (1, 32, 56, 88, 248, 704)
        elif "eca_nfnet_l0" in cfg.backbone:
            encoder_channels = (1, 64, 256, 512, 1536, 2304)
            backbone_out = 2304
        else:
            backbone_out = 2048

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=cfg.seg_dim,
            activation=None,
            kernel_size=3,
        )

        if cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif cfg.pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=self.cfg.gem_p_trainable)

        self.dropout = nn.Dropout(cfg.drop_rate)

        self.fc_head = nn.Linear(backbone_out, cfg.n_classes)

        if self.cfg.class_weights is not None:
            weight = torch.Tensor(self.cfg.class_weights).to("cuda")
        else:
            weight = None

        if self.cfg.loss == "BCE":
            self.class_loss = nn.BCEWithLogitsLoss()
        elif self.cfg.loss == "CE":
            self.class_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.class_loss = DenseCrossEntropy()

        self.seg_loss = nn.BCEWithLogitsLoss(reduction="mean")

        self.w = cfg.seg_weight

    def forward(self, batch):
        images = batch["input"]

        enc_out = self.encoder(images)
        x = enc_out[-1]
        x = self.global_pool(x)[:, :, 0, 0]
        x = self.dropout(x)

        logits = self.fc_head(x)

        target = batch["target"].float()
        if self.cfg.loss == "CE":
            target = target.argmax(dim=1).long()

        cls_loss = self.class_loss(logits, target)

        if batch["is_annotated"].sum() > 0:
            ia = batch["is_annotated"] > 0

            decoder_out = self.decoder(*[images] + enc_out)

            x_seg = self.segmentation_head(decoder_out)

            seg_loss = self.seg_loss(x_seg[ia].permute(0, 2, 3, 1), batch["mask"][ia])

        else:
            seg_loss = torch.zeros_like(cls_loss)

        loss = cls_loss + self.w * seg_loss

        output = {
            "loss": loss,
            "class_loss2": cls_loss,
            "seg_loss2": seg_loss,
            "class_logits": logits,
            "class_target": batch["target"].float(),
        }

        output["study_idx"] = batch["study_idx"]

        return output
