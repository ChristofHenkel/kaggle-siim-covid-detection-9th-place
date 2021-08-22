from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
import random
import os
import numpy as np
import pandas as pd

from torch.utils.data import SequentialSampler, DataLoader
import torch
from torch import optim
import importlib


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_model(cfg, train_dataset):
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)
    if cfg.pretrained_weights is not None:
        state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")["model"]
        if cfg.pretrained_weights_strict == False:
            print("popping head")
            to_pop = []
            for key in state_dict:
                if "head" in key:
                    to_pop += [key]
            for key in to_pop:
                print(f"popping {key}")
                state_dict.pop(key)

        net.load_state_dict(state_dict, strict=cfg.pretrained_weights_strict)
        print("weights loaded from", cfg.pretrained_weights)

    return net


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def get_train_dataset(train_df, cfg):
    print("Loading train dataset")

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    return train_dataset


def get_train_dataloader(train_ds, cfg):

    sampler = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_test_dataset(test_df, cfg):
    print("Loading test dataset")
    test_dataset = cfg.CustomDataset(test_df, cfg, aug=cfg.val_aug, mode="test")
    return test_dataset


def get_test_dataloader(test_ds, cfg):

    sampler = SequentialSampler(test_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader


def get_optimizer(model, cfg):
    params = model.parameters()

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return optimizer


def get_scheduler(cfg, optimizer, total_steps):

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size) // cfg.world_size,
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size) // cfg.world_size,
    )

    return scheduler


def get_data(cfg):
    if ".csv" in cfg.train_df:
        print(f"reading {cfg.train_df}")
        df = pd.read_csv(cfg.train_df)
    elif ".pq" in cfg.train_df:
        df = pd.read_parquet(cfg.train_df)
    else:
        df = pd.read_csv(cfg.train_df)

    if cfg.data_sample > -1:
        df = df.sample(cfg.data_sample)

    if cfg.val_df is not None:
        print(f"reading {cfg.val_df}")
        val_df = pd.read_csv(cfg.val_df)
        if cfg.val_fold > -1:
            val_df = val_df[val_df["fold"] == cfg.val_fold]

    train_df = df[df["fold"] != cfg.fold]
    if cfg.val_df is None:
        if cfg.val_fold > -1:
            val_df = df[df["fold"] == cfg.val_fold]
        else:
            val_df = df[df["fold"] == cfg.fold]

    if cfg.test:
        test_df = pd.read_csv(cfg.test_df)
        return train_df, val_df, test_df

    return train_df, val_df
