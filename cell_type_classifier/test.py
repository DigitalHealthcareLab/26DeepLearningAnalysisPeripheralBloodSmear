import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models import load_ckpt
from train import load_feats, l2norm

CLASS_NAMES = [
    "Neutrophil",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Basophil",
    "Immature cell",
    "nRBC",
    "Non-eligible",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_feats", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def infer_probs(model, X, device, batch_size):
    xb = torch.from_numpy(l2norm(X)).float()
    out = []
    for i in range(0, len(xb), batch_size):
        logits = model(xb[i:i + batch_size].to(device))
        out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


def merge_non_eligible(y, probs):
    y = y.copy()
    y[y == 8] = 7
    probs = probs.copy()
    probs[:, 7] = probs[:, 7] + probs[:, 8]
    probs = probs[:, :8]
    probs = probs / probs.sum(axis=1, keepdims=True)
    return y, probs


def per_class_auroc(y, probs):
    onehot = np.eye(probs.shape[1])[y]
    aurocs = []
    for c in range(probs.shape[1]):
        try:
            aurocs.append(roc_auc_score(onehot[:, c], probs[:, c]))
        except ValueError:
            aurocs.append(np.nan)
    return np.array(aurocs)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_ckpt(args.ckpt, device)

    X, y = load_feats(args.test_feats)
    y, probs = merge_non_eligible(y, infer_probs(model, X, device, args.batch_size))

    point = per_class_auroc(y, probs)
    rng = np.random.default_rng(args.seed)
    n = len(y)
    boot = []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, n, n)
        boot.append(per_class_auroc(y[idx], probs[idx]))
    lower, upper = np.nanpercentile(np.array(boot), [2.5, 97.5], axis=0)

    results = {
        name: {
            "auroc": float(point[c]),
            "auroc_ci_lower": float(lower[c]),
            "auroc_ci_upper": float(upper[c]),
        }
        for c, name in enumerate(CLASS_NAMES)
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
