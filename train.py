import numpy as np
import importlib
import sys
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from timm.utils.agc import adaptive_clip_grad
from collections import defaultdict

from utils import (
    set_seed,
    get_model,
    create_checkpoint,
    get_train_dataset,
    get_data,
)
from utils import (
    get_train_dataloader,
    get_test_dataset,
    get_test_dataloader,
    get_optimizer,
    get_scheduler,
)

import cv2
from copy import copy
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cv2.setNumThreads(0)

sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("losses")
sys.path.append("utils")


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

# overwrite params in config with additional args
if len(other_args) > 1:
    other_args = {k.replace("-", ""): v for k, v in zip(other_args[1::2], other_args[2::2])}

    for key in other_args:
        if key in cfg.__dict__:

            print(f"overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}")
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == "True"
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])

if cfg.test:
    from submission import create_submission

os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device


def run_predict_for_submission(model, test_dataloader, test_df, cfg, pre="test"):

    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    test_data = defaultdict(list)

    for data in tqdm(test_dataloader, disable=cfg.local_rank != 0):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        for key, test in output.items():
            test_data[key] += [output[key]]

    for key, val in output.items():
        value = test_data[key]
        if isinstance(value[0], list):
            test_data[key] = [item for sublist in value for item in sublist]

        else:
            if len(value[0].shape) == 0:
                test_data[key] = torch.stack(value)
            else:
                test_data[key] = torch.cat(value, dim=0)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            torch.save(test_data, f"{cfg.output_dir}/fold{cfg.fold}/{pre}_data_seed{cfg.seed}.pth")

    if cfg.local_rank == 0 and cfg.create_submission:

        create_submission(cfg, test_data, test_dataloader.dataset, pre)

    print("TEST FINISHED")


if __name__ == "__main__":

    # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    cfg.local_rank = 0
    cfg.world_size = 1
    cfg.rank = 0  # global rank

    device = "cuda:%d" % cfg.gpu
    cfg.device = device

    set_seed(cfg.seed)

    if cfg.test:
        train_df, val_df, test_df = get_data(cfg)
    else:
        train_df, val_df = get_data(cfg)

    train_dataset = get_train_dataset(train_df, cfg)
    if cfg.test:
        test_dataset = get_test_dataset(test_df, cfg)

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    if cfg.test:
        test_dataloader = get_test_dataloader(test_dataset, cfg)

    model = get_model(cfg, train_dataset)
    model.to(device)

    # if cfg.pretrained_weights is not None:
    #     model.load_state_dict(torch.load(cfg.pretrained_weights, map_location='cpu')['model'], strict=True)
    #     print('weights loaded from',cfg.pretrained_weights)

    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()

    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch + cfg.local_rank)

        cfg.curr_epoch = epoch

        print("EPOCH:", epoch)
        if cfg.reload_train_loader:
            train_dataset = get_train_dataset(train_df, cfg)
            train_dataloader = get_train_dataloader(train_dataset, cfg)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                cfg.curr_step += cfg.batch_size * cfg.world_size

                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")
                    # continue

                model.train()
                torch.set_grad_enabled(True)

                # Forward pass

                batch = batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                # Backward pass

                if cfg.optimizer == "SAM":

                    def closure():
                        optimizer.zero_grad()
                        output = model(batch)
                        loss = output["loss"]
                        loss.backward()
                        return loss

                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    if cfg.clip_grad > 0:
                        scaler.unscale_(optimizer)
                        if cfg.clip_mode == "norm":
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                        elif cfg.clip_mode == "agc":
                            adaptive_clip_grad(model.parameters(), cfg.clip_grad, norm_type=2.0)
                    if i % cfg.grad_accumulation == 0:
                        if cfg.optimizer == "SAM":
                            scaler.step(optimizer, closure)
                        else:
                            scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if cfg.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    if i % cfg.grad_accumulation == 0:
                        if cfg.optimizer == "SAM":
                            optimizer.step(closure)
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if cfg.local_rank == 0 and cfg.curr_step % cfg.batch_size == 0:
                    loss_names = [key for key in output_dict if "loss" in key]

                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

        if cfg.local_rank == 0 and cfg.epochs > 0:
            # print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler
            )

            torch.save(
                checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
            )

    if cfg.local_rank == 0 and cfg.epochs > 0:
        # print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
        checkpoint = create_checkpoint(model, optimizer, epoch, scheduler=scheduler, scaler=scaler)

        torch.save(
            checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
        )

    if cfg.test:
        run_predict_for_submission(model, test_dataloader, test_df, cfg, pre="test")
