import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import ast
import pandas as pd
import random


def batch_to_device(batch, device):
    batch_dict = {
        key: batch[key].to(device)
        for key in ["target", "input", "study_idx", "mask", "is_annotated"]
    }
    batch_dict["boxes"] = [item.to(device) for item in batch["boxes"]]
    return batch_dict


def custom_collate(batch):

    batch_dict = {}

    for key in ["target", "input", "study_idx", "mask", "is_annotated"]:
        batch_dict[key] = torch.stack([b[key] for b in batch])

    batch_dict["boxes"] = [b["boxes"] for b in batch]
    return batch_dict


tr_collate_fn = custom_collate
val_collate_fn = custom_collate


def rand_bbox(size, lamb):
    """ Generate random bounding box
    Args:
        - size: [width, breadth] of the bounding box
        - lamb: (lambda) cut ratio parameter
    Returns:
        - Bounding box
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1.0 - lamb)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.df = df.copy().reset_index(drop=True)

        self.image_ids = self.df["id"].str.replace("_image", "").values
        self.study_ids = self.df["StudyInstanceUID"].values
        self.data_folder = cfg.data_folder

        if mode == "test":
            self.df["boxes"] = "[]"
            self.df[cfg.classes] = 0
            self.data_folder = cfg.test_data_folder

        self.box_strs = self.df["boxes"].fillna("[]").values

        self.boxes = [self.get_img_boxes(box_str) for box_str in self.box_strs]
        self.wh = self.df[["width", "height"]].values * self.cfg.scale

        self.study_id2idx = {
            item: i for i, item in enumerate(self.df["StudyInstanceUID"].unique())
        }
        self.study_idxs = self.df["StudyInstanceUID"].map(self.study_id2idx).values

        self.series_ids = self.df["series_id"].values
        self.labels = self.df[cfg.classes].values
        self.normalization = cfg.normalization
        self.mode = mode
        self.aug = aug

    def get_item(self, idx):

        image_id = self.image_ids[idx]
        study_id = self.study_ids[idx]
        series_id = self.series_ids[idx]
        study_idx = self.study_idxs[idx]
        label = self.labels[idx]
        boxes = self.boxes[idx]
        w, h = self.wh[idx]

        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0].clip(0, w - 1)
            boxes[:, 1] = boxes[:, 1].clip(0, h - 1)
            boxes[:, 2] = boxes[:, 2].clip(0, w - 1)
            boxes[:, 3] = boxes[:, 3].clip(0, h - 1)

        fp = f"{self.data_folder}/{study_id}/{series_id}/{image_id}{self.cfg.suffix}"

        img = self.load_one(fp)

        if self.aug:
            img, boxes = self.augment(img, boxes)

        return img, label, boxes, study_idx

    def mixup(self, img, label, boxes):

        if self.cfg.mix_same:
            rnd_idx = np.random.choice(
                self.df[self.df[self.cfg.classes[label.argmax()]] == 1].index.values
            )
        else:
            rnd_idx = np.random.choice(np.arange(len(self)))

        rnd_img, rnd_label, rnd_boxes, _ = self.get_item(rnd_idx)

        lam = np.random.beta(1, 1)

        img = lam * img + (1 - lam) * rnd_img
        if self.cfg.mixadd:
            label = (label + rnd_label).clip(0, 1)
        else:
            label = lam * label + (1 - lam) * rnd_label
        if len(boxes) > 0 and len(rnd_boxes) > 0:
            boxes = np.vstack([boxes, rnd_boxes])
        elif len(rnd_boxes) > 0:
            boxes = rnd_boxes

        return img, label, boxes

    def cutmix(self, img, label, boxes, beta=1.0, th=0.25):

        # generate mixed sample
        fill_value = 255
        lam = np.random.beta(beta, beta)

        if self.cfg.mix_same:
            rnd_idx = np.random.choice(
                self.df[self.df[self.cfg.classes[label.argmax()]] == 1].index.values
            )
        else:
            rnd_idx = np.random.choice(np.arange(len(self)))

        rnd_img, rnd_label, rnd_boxes, _ = self.get_item(rnd_idx)

        w, h, c = img.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox((w, h, c), lam)
        cutmix_image = img.copy()
        cutmix_image[bby1:bby2, bbx1:bbx2, :] = fill_value

        # bboxes
        new_bboxes = []
        if len(boxes) > 0:
            for bbox in boxes:
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
                overlapping_size = np.sum(
                    (cutmix_image[y_min:y_max, x_min:x_max, 0] == fill_value)
                )

                # Add the bbox if it has less than some threshold of content is inside the cutout patch
                if overlapping_size / bbox_size < th:
                    new_bboxes.append(bbox)

        mask = np.zeros(rnd_img.shape)
        mask[bby1:bby2, bbx1:bbx2, :] = 1
        image2 = rnd_img * mask

        if len(rnd_boxes) > 0:
            for bbox in rnd_boxes:
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
                overlapping_size = np.sum((image2[y_min:y_max, x_min:x_max, 0] == 0))

                # Add the bbox if it has less than some threshold of content is inside the cutout patch
                if overlapping_size / bbox_size < th:
                    new_bboxes.append(bbox)

        cutmix_image[bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]

        if self.cfg.mixadd:
            label = (label + rnd_label).clip(0, 1)
        else:
            label = lam * label + (1 - lam) * rnd_label

        return cutmix_image, label, np.array(new_bboxes)

    def mosaic(self, image, label, boxes, imsize=512):

        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [
            int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)
        ]  # center x, y
        indexes = [-1] + [np.random.choice(np.arange(len(self))) for _ in range(3)]

        result_image = np.full((imsize, imsize, 1), 1, dtype=np.float32)
        result_boxes = []

        labels = 0

        for i, index in enumerate(indexes):
            if i > 0:
                image, label, boxes, _ = self.get_item(index)
            if i == 0:
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            area = ((y2b - y1b) * (x2b - x1b)) / (imsize * imsize)

            if self.cfg.mixadd:
                labels += label
            else:
                labels += area * label

            if len(boxes) > 0:
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)

        if len(result_boxes) > 0:
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            result_boxes = result_boxes[
                np.where(
                    (result_boxes[:, 2] - result_boxes[:, 0])
                    * (result_boxes[:, 3] - result_boxes[:, 1])
                    > 0
                )
            ]
        else:
            result_boxes = []

        labels = labels.clip(0, 1)

        return result_image, labels, result_boxes

    def boxmix(self, image, label, boxes):
        rnd_idx = np.random.choice(np.arange(len(self)))

        rnd_img, rnd_label, rnd_boxes, _ = self.get_item(rnd_idx)

        if len(boxes) > 0:
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                image[y_min:y_max, x_min:x_max, :] = rnd_img[y_min:y_max, x_min:x_max, :]

        if len(rnd_boxes) > 0:
            for box in rnd_boxes:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                image[y_min:y_max, x_min:x_max, :] = rnd_img[y_min:y_max, x_min:x_max, :]

        result_boxes = rnd_boxes
        result_label = rnd_label

        return image, result_label, result_boxes

    def halfmix(self, image, label, boxes):
        max_size = image.shape[1]
        half = max_size // 2

        result_boxes = []
        if np.random.random() <= 0.5:
            image[:, half:, :] = 0
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                if x_min <= half and x_max <= half * 1.2:
                    result_boxes.append(box)
        else:
            image[:, :half, :] = 0
            for box in boxes:
                x_min, y_min, x_max, y_max = map(int, box[:4])
                if x_min >= half * 0.8 and x_max >= half:
                    result_boxes.append(box)

        result_boxes = []

        if len(result_boxes) > 0:
            result_boxes = np.stack(result_boxes)
        else:
            result_boxes = []

        return image, label, boxes

    def __getitem__(self, idx):

        img, label, boxes, study_idx = self.get_item(idx)

        if self.mode == "train":
            if np.random.random() <= self.cfg.mixup:
                img, label, boxes = self.mixup(img, label, boxes)
            if np.random.random() <= self.cfg.cutmix:
                img, label, boxes = self.cutmix(img, label, boxes)
            if np.random.random() <= self.cfg.mosaic:
                img, label, boxes = self.mosaic(img, label, boxes)
            if np.random.random() <= self.cfg.boxmix and label.argmax() == 0:
                img, label, boxes = self.boxmix(img, label, boxes)
            if np.random.random() <= self.cfg.halfmix:
                img, label, boxes = self.halfmix(img, label, boxes)

            if self.cfg.label_smoothing > 0:
                label = label.clip(self.cfg.label_smoothing, 1 - (self.cfg.label_smoothing * 3))

        if self.cfg.normalization is not None:
            img = self.normalize_img(img)

        if len(boxes) > 0:
            boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

        if self.cfg.seg_dim > 1:
            mask = np.zeros((img.shape[0], img.shape[1], self.cfg.seg_dim))
            label_idx = np.argmax(label)
        else:
            mask = np.zeros_like(img)
            label_idx = 0

        if len(boxes) > 0:
            for box in boxes:
                mask[int(box[0]) : int(box[2]), int(box[1]) : int(box[3]), label_idx] = 1
            is_annotated = 1
        else:
            is_annotated = 0

        feature_dict = {
            "input": torch.tensor(img.transpose(2, 0, 1)),
            "mask": torch.tensor(mask),
            "target": torch.tensor(label),
            "boxes": torch.tensor(boxes),
            "study_idx": torch.tensor(study_idx),
            "is_annotated": torch.tensor(is_annotated),
        }
        return feature_dict

    def get_img_boxes(self, box_str):
        b = ast.literal_eval(box_str)
        boxes = np.zeros((len(b), 5))
        if len(b) > 0:
            b = pd.DataFrame(b)[["x", "y", "width", "height"]].values
            b[:, 2:] += b[:, :2]
            b = b * self.cfg.scale
            boxes[:, :4] = b
        #         else:
        #             b = np.array([[],[],[],[],[]])
        return boxes

    def __len__(self):
        return len(self.image_ids)

    def augment(self, img, boxes):
        transformed = self.aug(image=img, bboxes=boxes)
        img_aug = transformed["image"]
        boxes_aug = transformed["bboxes"]
        boxes_aug = np.array(boxes_aug)
        return img_aug, boxes_aug

    def load_one(self, fp):

        try:
            if self.cfg.suffix == ".dcm":
                img = self.load_dicom(fp)
            else:
                img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                img = img[:, :, None]
                # .astype(np.float32)
        except:
            print("FAIL READING img", fp)

        return img

    def load_dicom(self, fn):
        pass

    def normalize_img(self, img):

        if self.normalization == "image":
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20, 20)

        elif self.normalization == "simple":
            img = img / 255

        elif self.normalization == "min_max":
            img = img - np.min(img)
            img = img / np.max(img)

        return img.astype(np.float32)
