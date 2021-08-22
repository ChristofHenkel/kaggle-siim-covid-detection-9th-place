import torch
from torch import nn
import timm
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Beta
from segmentation_models_pytorch.unet.model import SegmentationHead
import math


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


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Y2=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            if len(Y.shape) == 1:
                Y = coeffs * Y + (1 - coeffs) * Y[perm]
            else:
                Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if Y2 is not None:
            if self.mixadd:
                Y2 = (Y2 + Y2[perm]).clip(0, 1)
            else:
                n_dims_y2 = len(Y2.shape)
                if n_dims_y2 == 2:
                    Y2 = coeffs.view(-1, 1) * Y2 + (1 - coeffs.view(-1, 1)) * Y2[perm]
                elif n_dims_y2 == 3:
                    Y2 = coeffs.view(-1, 1, 1) * Y2 + (1 - coeffs.view(-1, 1, 1)) * Y2[perm]
                else:
                    Y2 = coeffs.view(-1, 1, 1, 1) * Y2 + (1 - coeffs.view(-1, 1, 1, 1)) * Y2[perm]
            return X, Y, Y2

        return X, Y


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
            num_classes=0,
            in_chans=1,
            drop_path_rate=cfg.drop_path_rate,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
        )

        self.cfg = cfg

        if "xcit_small_24_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_24_p16" in cfg.backbone:
            backbone_out = 512
        elif "xcit_small_12_p16" in cfg.backbone:
            backbone_out = 384
        elif "xcit_medium_12_p16" in cfg.backbone:
            backbone_out = 512
        elif "xcit_tiny_24_p8" in cfg.backbone:
            backbone_out = 192
        else:
            backbone_out = 2048

        self.segmentation_head = SegmentationHead(
            in_channels=backbone_out, out_channels=cfg.seg_dim, activation=None, kernel_size=3,
        )

        self.backbone_out = backbone_out
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

        x = batch["input"]
        masks = batch["mask"]
        target = batch["target"].float()

        B = x.shape[0]
        # x is (B, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        x, (Hp, Wp) = self.encoder.patch_embed(x)

        if self.encoder.use_pos_embed:
            # `pos_embed` (B, C, Hp, Wp), reshape -> (B, C, N), permute -> (B, N, C)
            pos_encoding = (
                self.encoder.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            )
            x = x + pos_encoding

        x = self.encoder.pos_drop(x)

        for blk in self.encoder.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder.cls_attn_blocks:
            x = blk(x)

        x_pooled = self.encoder.norm(x)[:, 0]
        x_pooled = self.dropout(x_pooled)

        logits = self.fc_head(x_pooled)

        cls_loss = self.class_loss(logits, target)

        sz = int(math.sqrt(x.shape[1] - 1))
        x = x[:, 1:].reshape(B, sz, sz, self.backbone_out)
        x = x.permute(0, 3, 1, 2)
        x_seg = self.segmentation_head(x)

        masks = F.interpolate(masks.permute(0, 3, 1, 2), (x_seg.shape[2], x_seg.shape[3]))
        seg_loss = self.seg_loss(x_seg, masks)

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
