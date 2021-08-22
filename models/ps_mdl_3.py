from effdet import get_efficientdet_config
import torch
from torch import nn
from typing import Dict
from effdet.anchors import Anchors, AnchorLabeler
from effdet.loss import DetectionLoss
from effdet.bench import _batch_detection, _post_process
from effdet.factory import create_model_from_config
import timm
from torch.nn import functional as F
from torch.nn.parameter import Parameter


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


class DetBenchTrain(nn.Module):
    def __init__(self, model, create_labeler=True):
        super(DetBenchTrain, self).__init__()
        self.model = model
        self.config = (
            model.config
        )  # FIXME remove this when we can use @property (torchscript limitation)
        self.num_levels = model.config.num_levels
        self.num_classes = model.config.num_classes
        self.anchors = Anchors.from_config(model.config)
        self.max_detection_points = model.config.max_detection_points
        self.max_det_per_image = model.config.max_det_per_image
        self.soft_nms = model.config.soft_nms
        self.anchor_labeler = None
        if create_labeler:
            self.anchor_labeler = AnchorLabeler(
                self.anchors, self.num_classes, match_threshold=0.5
            )
        self.loss_fn = DetectionLoss(model.config)

    def forward(self, x, target: Dict[str, torch.Tensor]):
        class_out, box_out = self.model(x)
        if self.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert "label_num_positives" in target
            cls_targets = [target[f"label_cls_{l}"] for l in range(self.num_levels)]
            box_targets = [target[f"label_bbox_{l}"] for l in range(self.num_levels)]
            num_positives = target["label_num_positives"]
        else:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                target["bbox"], target["cls"]
            )

        loss, class_loss, box_loss = self.loss_fn(
            class_out, box_out, cls_targets, box_targets, num_positives
        )
        output = {"loss": loss, "cls_loss": class_loss, "box_loss": box_loss}
        if not self.training:
            # if eval mode, output detections for evaluation
            class_out_pp, box_out_pp, indices, classes = _post_process(
                class_out, box_out, num_levels=self.num_levels, num_classes=self.num_classes
            )
            output["detections"] = _batch_detection(
                x.shape[0], class_out_pp, box_out_pp, self.anchors.boxes, indices, classes
            )
        return output


def get_backbone(cfg):
    config = get_efficientdet_config(cfg.backbone)
    config.num_classes = 1
    config.image_size = cfg.image_size

    config.backbone_args.in_chans = cfg.in_channels

    net = create_model_from_config(
        config, pretrained_backbone=cfg.pretrained
    )

    model = DetBenchTrain(net)

    return model, config


class ConvHead(nn.Module):
    def __init__(self, config, cfg):
        super(ConvHead, self).__init__()
        self.backbone = timm.create_model(config.backbone_name, pretrained=cfg.pretrained_convhead)
        del self.backbone.conv_stem
        del self.backbone.blocks
        del self.backbone.bn1
        del self.backbone.act1
        del self.backbone.global_pool
        del self.backbone.classifier

    def forward(self, x):
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        return x


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.effdet, config = get_backbone(cfg)

        self.conv_head = ConvHead(config, cfg)

        # print(self.backbone.config)
        self.cfg = cfg
        if cfg.pretrained_weights is not None:
            self.load_state_dict(
                torch.load(cfg.pretrained_weights, map_location="cpu"), strict=False
            )
            print("weights loaded from", cfg.pretrained_weights)

        if cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=self.cfg.gem_p_trainable)

        backbone_out = self.conv_head.backbone.num_features

        self.fc_head = nn.Linear(backbone_out, cfg.n_classes)

        if self.cfg.loss == "BCE":
            self.class_loss = nn.BCEWithLogitsLoss()
        else:
            self.class_loss = DenseCrossEntropy()

    def forward(self, batch):
        images = batch["input"]
        label = batch["target"].argmax(dim=1)

        target = {}
        bbox = []
        cls_ = []
        cnt = 0
        for b, c in zip(batch["boxes"], label):
            if len(b) > 0:
                bbox += [b[:, :4].float()]
                cls_ += [b[:, -1] + 1]
            else:
                cnt += 1
                bbox += [torch.tensor([[], [], [], []], device=images.device).permute(1, 0)]
                cls_ += [torch.tensor([], device=images.device)]

        target["cls"] = cls_
        target["bbox"] = bbox

        x_out = self.effdet.model.backbone(images)
        x = self.effdet.model.fpn(x_out)
        class_out = self.effdet.model.class_net(x)
        box_out = self.effdet.model.box_net(x)

        x_out = self.conv_head(x_out[2])

        x_classes = self.global_pool(x_out)
        x_classes = x_classes[:, :, 0, 0]

        x_classes = self.fc_head(x_classes)

        if self.effdet.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert "label_num_positives" in target
            cls_targets = [target[f"label_cls_{l}"] for l in range(self.effdet.num_levels)]
            box_targets = [target[f"label_bbox_{l}"] for l in range(self.effdet.num_levels)]
            num_positives = target["label_num_positives"]
        else:
            (
                cls_targets,
                box_targets,
                num_positives,
            ) = self.effdet.anchor_labeler.batch_label_anchors(target["bbox"], target["cls"])

        loss, class_loss, box_loss = self.effdet.loss_fn(
            class_out, box_out, cls_targets, box_targets, num_positives
        )
        class_loss2 = self.class_loss(x_classes, batch["target"].float())

        loss = loss + self.cfg.study_weight * class_loss2

        output = {
            "loss": loss,
            "class_loss": class_loss,
            "box_loss": box_loss,
            "class_loss2": class_loss2,
            "class_logits": x_classes,
            "class_target": batch["target"].float(),
            "box_target": target["bbox"],
        }
        if not self.training:
            class_out_pp, box_out_pp, indices, classes = _post_process(
                class_out,
                box_out,
                num_levels=self.effdet.num_levels,
                num_classes=self.effdet.num_classes,
                # max_detection_points=self.backbone.max_detection_points
            )

            output["detections"] = _batch_detection(
                images.shape[0],
                class_out_pp,
                box_out_pp,
                self.effdet.anchors.boxes,
                indices,
                classes,
            )

        output["study_idx"] = batch["study_idx"]

        return output
