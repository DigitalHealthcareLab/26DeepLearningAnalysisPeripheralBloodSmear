import argparse
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from configs import ExperimentConfig
from models.encoder import PBSEncoder
from models.pool import PMAPool
from models.heads import DownstreamHead
from models.pbsmodel import PBSModel

from dataset import PBSDSDataset, load_anno_df
from collate import collate_fn
from trainer import Trainer
from utils import (
    set_seed,
    build_criterion,
    build_metrics,
)


def create_config(args):
    overrides = {}

    if args.config_file:
        exp_cfg = ExperimentConfig.load(args.config_file)
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        config_name = Path(args.config_file).stem + "_" + timestamp
        overrides["logging.log_dir"] = os.path.join(
            exp_cfg.logging.log_dir, args.experiment_name, config_name,
        )
    else:
        exp_cfg = ExperimentConfig()
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        overrides["logging.log_dir"] = os.path.join(
            exp_cfg.logging.log_dir, args.experiment_name, timestamp,
        )

    if args.root_path:
        overrides["data.root_path"] = args.root_path
    if args.anno_path:
        overrides["data.anno_path"] = os.path.join(args.anno_path, args.experiment_name)
    if args.remove_type is not None:
        overrides["data.remove_type"] = args.remove_type

    if args.device:
        overrides["runtime.device"] = args.device

    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.task is not None:
        overrides["encoder.task"] = args.task
    if args.num_classes is not None:
        overrides["encoder.num_classes"] = args.num_classes

    exp_cfg.override_from_dict(overrides)
    return exp_cfg


def main(exp_cfg):
    set_seed(exp_cfg.runtime.seed, exp_cfg.runtime.deterministic)
    device = torch.device(exp_cfg.runtime.device)

    task = exp_cfg.encoder.task
    if task == "multiclass":
        exp_cfg.data.label_dtype = torch.long
    else:
        exp_cfg.data.label_dtype = torch.float32

    num_classes = exp_cfg.encoder.num_classes

    log_dir = exp_cfg.logging.log_dir

    train_anno_path = os.path.join(exp_cfg.data.anno_path, "train.csv")
    valid_anno_path = os.path.join(exp_cfg.data.anno_path, "valid.csv")
    test_anno_path = os.path.join(exp_cfg.data.anno_path, "test.csv")
    test_h_anno_path = os.path.join(exp_cfg.data.anno_path, "test_h.csv")

    train_df = load_anno_df(train_anno_path, exp_cfg.data)
    val_df = load_anno_df(valid_anno_path, exp_cfg.data)
    test_df = load_anno_df(test_anno_path, exp_cfg.data)
    test_h_df = load_anno_df(test_h_anno_path, exp_cfg.data)

    train_dataset = PBSDSDataset(exp_cfg.data, train_df)
    val_dataset = PBSDSDataset(exp_cfg.data, val_df)
    test_dataset = PBSDSDataset(exp_cfg.data, test_df)
    test_h_dataset = PBSDSDataset(exp_cfg.data, test_h_df)

    encoder = PBSEncoder(exp_cfg.encoder).to(device)
    pool = PMAPool(
        embed_dim=exp_cfg.encoder.d_model,
        n_heads=exp_cfg.encoder.n_heads,
        attn_dropout=0.0,
        dropout=exp_cfg.encoder.pool_dropout,
    ).to(device)
    head = DownstreamHead(
        d_model=exp_cfg.encoder.d_model,
        num_classes=num_classes,
        dropout=exp_cfg.encoder.dropout,
    ).to(device)

    model = PBSModel(encoder=encoder, pool=pool, head=head).to(device)

    criterion = build_criterion(
        task=task,
        df_train=train_df,
        num_classes=num_classes,
        device=device,
    )
    metrics, primary_key = build_metrics(
        task=task, num_classes=num_classes,
    )

    collate = partial(
        collate_fn,
        pad_token_id=exp_cfg.encoder.type_vocab_size,
    )
    bs = exp_cfg.train.batch_size
    nw = exp_cfg.train.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        persistent_workers=(nw > 0),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
        collate_fn=collate,
    )

    test_h_loader = DataLoader(
        test_h_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
        collate_fn=collate,
    )

    param_groups = model.get_param_groups(
        lr=exp_cfg.train.lr,
        encoder_lr_scale=0.01,
        pma_lr_scale=0.1,
        weight_decay=exp_cfg.train.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    warmup = LinearLR(
        optimizer,
        start_factor=exp_cfg.train.start_factor,
        end_factor=1.0,
        total_iters=exp_cfg.train.warmup_epochs,
    )
    main_sched = CosineAnnealingLR(
        optimizer,
        T_max=exp_cfg.train.epochs - exp_cfg.train.warmup_epochs,
        eta_min=exp_cfg.train.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, main_sched],
        milestones=[exp_cfg.train.warmup_epochs],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_h_loader=test_h_loader,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task=task,
        primary_key=primary_key,
        log_dir=log_dir,
        patience=exp_cfg.train.patience,
    )

    trainer.fit(epochs=exp_cfg.train.epochs)

    trainer.run_bootstrap_test(n_bootstrap=1000, seed=exp_cfg.runtime.seed)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str,
                        default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--anno_path", type=str,
                        default=None)
    parser.add_argument("--remove_type", type=int, default=None, nargs="+")

    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--task", type=str, default=None,
                        choices=["binary", "multiclass"])
    parser.add_argument("--num_classes", type=int, default=None)


    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exp_cfg = create_config(args)

    exp_root = exp_cfg.logging.log_dir
    exp_cfg.save_config(exp_root)

    main(exp_cfg=exp_cfg)
