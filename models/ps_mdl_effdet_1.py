from effdet import get_efficientdet_config
import torch
from torch import nn
from typing import Dict
from effdet.anchors import Anchors, AnchorLabeler
from effdet.loss import DetectionLoss
from effdet.bench import _batch_detection, _post_process
from effdet.factory import create_model_from_config
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

    overrides = (
        "image_size",
        "box_loss_weight",
        "num_scales",
        "aspect_ratios",
    )
    for ov in overrides:
        setattr(config, ov, getattr(cfg, ov))

    config.backbone_args.in_chans = cfg.in_channels

    net = create_model_from_config(
        config, pretrained_backbone=cfg.pretrained
    )

    model = DetBenchTrain(net)

    return model, config


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.effdet, config = get_backbone(cfg)

        self.cfg = cfg
        if cfg.pretrained_weights is not None:
            self.load_state_dict(
                torch.load(cfg.pretrained_weights, map_location="cpu"), strict=False
            )
            print("weights loaded from", cfg.pretrained_weights)

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

        out = self.effdet(images, target)

        output = {
            "loss": out["loss"],
            "class_loss": out["cls_loss"],
            "box_loss": out["box_loss"],
            "box_target": target["bbox"],
        }
        if not self.training:

            output["detections"] = out["detections"]

        output["study_idx"] = batch["study_idx"]

        return output
