import argparse
import os

import torch

from models import load_ckpt
from train import l2norm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def predict_types(model, features, device):
    x = torch.from_numpy(l2norm(features.numpy())).float().to(device)
    return model(x).argmax(1).cpu()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_ckpt(args.ckpt, device)

    for sid in sorted(os.listdir(args.input_dir)):
        features = torch.load(
            os.path.join(args.input_dir, sid, "feats.pt"), map_location="cpu", weights_only=True
        )
        type_ids = predict_types(model, features, device)
        os.makedirs(os.path.join(args.output_dir, sid), exist_ok=True)
        torch.save((features, type_ids), os.path.join(args.output_dir, sid, "feats.pt"))


if __name__ == "__main__":
    main()
