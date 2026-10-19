import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.01, 0.1, 1.0, 10.0]
TOTAL_NUC_THRESHOLD = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--anno_path", type=str, required=True)
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["binary", "multiclass"])
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(anno_path, split):
    df = pd.read_csv(os.path.join(anno_path, f"{split}.csv"), dtype={"specimen_id": str})
    df = df[pd.to_numeric(df["total_nuc"], errors="coerce") >= TOTAL_NUC_THRESHOLD].reset_index(drop=True)
    X = np.column_stack([df["age"].to_numpy(np.float64),
                         (df["sex"].to_numpy() == "M").astype(np.float64)])
    return X, df["label"].to_numpy().astype(int), df["specimen_id"].to_numpy()


def auroc(y, s, multiclass):
    if multiclass:
        return roc_auc_score(y, s, multi_class="ovr", average="macro")
    return roc_auc_score(y, s)


def bootstrap_ci(y, s, multiclass, n_bootstrap, seed):
    rng = np.random.RandomState(seed)
    n, k = len(y), len(np.unique(y))
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < k:
            continue
        vals.append(auroc(y[idx], s[idx], multiclass))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_lr(X_tr, y_tr, X_val, y_val, multiclass):
    best_auroc, best_clf = -1.0, None
    for C in C_GRID:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
        ).fit(X_tr, y_tr)
        p = clf.predict_proba(X_val)
        val_auroc = auroc(y_val, p if multiclass else p[:, 1], multiclass)
        if val_auroc > best_auroc:
            best_auroc, best_clf = val_auroc, clf
    return best_clf


def main():
    args = parse_args()
    multiclass = args.task == "multiclass"
    os.makedirs(args.output_dir, exist_ok=True)
    anno_path = os.path.join(args.anno_path, args.experiment_name)

    X_tr, y_tr, _ = load_split(anno_path, "train")
    X_val, y_val, _ = load_split(anno_path, "valid")
    clf = fit_lr(X_tr, y_tr, X_val, y_val, multiclass)

    results = {}
    preds = {}
    for split in ("test", "test_h"):
        X, y, ids = load_split(anno_path, split)
        p = clf.predict_proba(X)
        s = p if multiclass else p[:, 1]
        lower, upper = bootstrap_ci(y, s, multiclass, args.n_bootstrap, args.seed)
        results[split] = {
            "auroc": float(auroc(y, s, multiclass)),
            "auroc_ci_lower": lower,
            "auroc_ci_upper": upper,
            "n_samples": len(y),
        }
        preds[f"{split}_probs"] = s.astype(np.float32)
        preds[f"{split}_labels"] = y
        preds[f"{split}_ids"] = ids.astype(str)

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(args.output_dir, "predictions.npz"), **preds)


if __name__ == "__main__":
    main()
