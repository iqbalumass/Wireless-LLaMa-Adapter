#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Adapted for Iqbal
"""

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_adapter import LLaMA_adapter

# Correct imports
from engine_finetune import train_one_epoch
from data.radar_dataset import RadarImageTextDataset, collate_fn

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path


# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser('Radar-LLaMA-Adapter fine-tuning (single GPU)', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=11, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str)
    parser.add_argument('--llama_path', default='/path/to/llama', type=str)
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str)
    parser.add_argument('--max_words', default=512, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=1)

    # Dataset and output
    parser.add_argument('--ann_file', default='./data/radar_annotations.json', type=str,
                        help='Path to radar–image–text JSON annotation file')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--log_dir', default='./output')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    return parser


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------
def main(args):
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n", json.dumps(vars(args), indent=2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # --------------------------
    # 1. Build model
    # --------------------------
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenizer_path = os.path.join(args.llama_path, 'tokenizer.model')

    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenizer_path)
    model.to(device)

    model_without_ddp = model  # DDP removed
    print("Model =", model_without_ddp)
    print("Trainable Params:")
    for key, val in model.named_parameters():
        if val.requires_grad:
            print(f"{key:60s} {tuple(val.shape)}")

    # --------------------------
    # 2. Optimizer setup
    # --------------------------
    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Base LR: {args.blr:.2e}, Actual LR: {args.lr:.2e}")
    print(f"Accumulate Grad Iterations: {args.accum_iter}")
    print(f"Effective Batch Size: {eff_batch_size}")

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("Optimizer:", optimizer)

    # scaler and pretrained weights
    loss_scaler = NativeScaler()
    misc.load_model(model_without_ddp, args.pretrained_path)

    # --------------------------
    # 3. Dataset and DataLoader
    # --------------------------
    dataset_train = RadarImageTextDataset(
        ann_file=args.ann_file,
        transform=model.clip_transform,
        tokenizer=model.tokenizer
    )
    print("Loaded dataset:", len(dataset_train), "samples")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn,   #add this
    )

    # Optional quick sanity check
    batch = next(iter(data_loader_train))
    print("Sample keys:", batch.keys())
    if "radar" in batch:
        print("Radar shape:", batch["radar"].shape)
    print("Image shape:", batch["image"].shape)

    # --------------------------
    # 4. Tensorboard logger
    # --------------------------
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # --------------------------
    # 5. Training Loop
    # --------------------------
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Save checkpoint
        # Save only last 2 epochs
        if args.output_dir:
            if epoch >= args.epochs - 2:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

        # Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            log_writer.flush()

    total_time = time.time() - start_time
    print('Training time:', str(datetime.timedelta(seconds=int(total_time))))


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
