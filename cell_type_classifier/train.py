import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models import MLP


def load_feats(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["feats"].numpy(), blob["labels"].numpy().astype(int)


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + 1e-8)


def to_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_feats", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3])
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, _ in loader:
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_mlp(X_tr, y_tr, X_val, y_val, lr, epochs, batch_size, n_classes,
              class_weights, weight_decay, device):
    model = MLP(X_tr.shape[1], 512, n_classes, 0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))

    tr_loader = to_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = to_loader(X_val, y_val, batch_size * 2, shuffle=False)

    best_val, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        val_acc = balanced_accuracy_score(y_val, predict(model, val_loader, device))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    X, y = load_feats(args.train_feats)
    X = l2norm(X)
    n_classes = 9
    counts = np.bincount(y, minlength=n_classes)
    class_weights = (counts.sum() / (n_classes * counts)).astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    best = None
    for lr in args.lrs:
        model, val_bal = train_mlp(
            X_tr, y_tr, X_val, y_val, lr=lr, epochs=args.epochs,
            batch_size=args.batch_size, n_classes=n_classes,
            class_weights=class_weights, weight_decay=args.weight_decay, device=device,
        )
        print(f"lr={lr:.0e}  val_balanced_acc={val_bal:.4f}")
        if best is None or val_bal > best["val_bal"]:
            best = {"model": model, "val_bal": val_bal}

    torch.save(best["model"].state_dict(), os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    main()
