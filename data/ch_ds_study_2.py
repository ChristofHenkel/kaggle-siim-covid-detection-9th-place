import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import ast
import pandas as pd


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
        try:
            self.labels = self.df[cfg.classes].values
        except:
            self.df[cfg.classes] = 0
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

        return img, label, boxes, study_idx

    def __getitem__(self, idx):

        img, label, boxes, study_idx = self.get_item(idx)

        mask = np.zeros_like(img)
        label_idx = 0

        if len(boxes) > 0:
            for box in boxes:
                mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2]), label_idx] = 1
            is_annotated = 1
        else:
            is_annotated = 0

        if self.aug:
            img, boxes, mask = self.augment(img, boxes, mask)

        if self.cfg.normalization is not None:
            img = self.normalize_img(img)

        if len(boxes) > 0:
            boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

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
        return boxes

    def __len__(self):
        return len(self.image_ids)

    def augment(self, img, boxes, mask):
        transformed = self.aug(image=img, bboxes=boxes, mask=mask)
        img_aug = transformed["image"]
        mask_aug = transformed["mask"]
        boxes_aug = transformed["bboxes"]
        boxes_aug = np.array(boxes_aug)
        return img_aug, boxes_aug, mask_aug.astype(np.float32)

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
        elif self.normalization == "simple0":
            img = 2 * (img / 255) - 1

        elif self.normalization == "min_max":
            img = img - np.min(img)
            img = img / np.max(img)

        return img.astype(np.float32)
