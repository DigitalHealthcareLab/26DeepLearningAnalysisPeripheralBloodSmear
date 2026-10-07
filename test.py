import argparse
import json
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from configs import ExperimentConfig
from models.encoder import PBSEncoder
from models.pool import PMAPool
from models.heads import DownstreamHead
from models.pbsmodel import PBSModel

from dataset import PBSDSDataset, load_anno_df
from collate import collate_fn
from utils import set_seed, test_with_bootstrap, format_test_json


def _build_loader(dataset, exp_cfg):
    collate = partial(
        collate_fn,
        pad_token_id=exp_cfg.encoder.type_vocab_size,
    )
    nw = exp_cfg.train.num_workers
    return DataLoader(
        dataset,
        batch_size=exp_cfg.train.batch_size,
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
        collate_fn=collate,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    config_path = os.path.join(log_dir, "config.yaml")
    best_ckpt_path = os.path.join(log_dir, "checkpoints", "best_model.pth")
    out_path = os.path.join(log_dir, "test_results.json")

    exp_cfg = ExperimentConfig.load(config_path)

    seed = exp_cfg.runtime.seed
    set_seed(seed, exp_cfg.runtime.deterministic)

    device = torch.device(exp_cfg.runtime.device)

    task = exp_cfg.encoder.task

    if task == "multiclass":
        exp_cfg.data.label_dtype = torch.long
    else:
        exp_cfg.data.label_dtype = torch.float32

    num_classes = exp_cfg.encoder.num_classes

    anno_path = exp_cfg.data.anno_path
    valid_anno_path = os.path.join(anno_path, "valid.csv")
    test_anno_path = os.path.join(anno_path, "test.csv")
    test_h_anno_path = os.path.join(anno_path, "test_h.csv")

    val_df = load_anno_df(valid_anno_path, exp_cfg.data)
    test_df = load_anno_df(test_anno_path, exp_cfg.data)
    test_h_df = load_anno_df(test_h_anno_path, exp_cfg.data)

    val_dataset = PBSDSDataset(exp_cfg.data, val_df)
    val_loader = _build_loader(val_dataset, exp_cfg)

    test_dataset = PBSDSDataset(exp_cfg.data, test_df)
    test_h_dataset = PBSDSDataset(exp_cfg.data, test_h_df)
    eval_loaders = [
        ("test", _build_loader(test_dataset, exp_cfg)),
        ("test_h", _build_loader(test_h_dataset, exp_cfg)),
    ]

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

    state = torch.load(best_ckpt_path, map_location=device, weights_only=True)

    model = PBSModel(encoder=encoder, pool=pool, head=head).to(device)

    model.load_state_dict_for_resume(state)
    model.eval()

    combined_json = {"checkpoint_path": best_ckpt_path}
    for name, loader in eval_loaders:
        result = test_with_bootstrap(
            model=model,
            loader=loader,
            device=device,
            task=task,
            n_bootstrap=1000,
            seed=seed,
            threshold_tuning_loader=val_loader if task == "binary" else None,
        )
        combined_json[name] = format_test_json(result)

    with open(out_path, "w") as f:
        json.dump(combined_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
