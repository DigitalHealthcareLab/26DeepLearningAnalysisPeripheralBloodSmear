import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_a", type=str, required=True)
    p.add_argument("--pred_b", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["binary", "multiclass"])
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--splits", type=str, nargs="+", default=["test", "test_h"])
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def auroc(y, s, multiclass):
    if multiclass:
        return roc_auc_score(y, s, multi_class="ovr", average="macro")
    return roc_auc_score(y, s)


def paired_bootstrap(y, s_a, s_b, multiclass, n_bootstrap, seed):
    rng = np.random.RandomState(seed)
    n, k = len(y), len(np.unique(y))
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < k:
            continue
        deltas.append(auroc(y[idx], s_a[idx], multiclass) - auroc(y[idx], s_b[idx], multiclass))
    deltas = np.asarray(deltas)
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    p = min(2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean()), 1.0)
    return float(lower), float(upper), float(p)


def main():
    args = parse_args()
    multiclass = args.task == "multiclass"
    pred_a, pred_b = np.load(args.pred_a), np.load(args.pred_b)

    results = {}
    for split in args.splits:
        assert np.array_equal(pred_a[f"{split}_ids"], pred_b[f"{split}_ids"])
        y = pred_a[f"{split}_labels"].astype(int)
        s_a, s_b = pred_a[f"{split}_probs"], pred_b[f"{split}_probs"]
        auroc_a, auroc_b = auroc(y, s_a, multiclass), auroc(y, s_b, multiclass)
        lower, upper, p = paired_bootstrap(y, s_a, s_b, multiclass, args.n_bootstrap, args.seed)
        results[split] = {
            "auroc_a": float(auroc_a),
            "auroc_b": float(auroc_b),
            "delta": float(auroc_a - auroc_b),
            "delta_ci_lower": lower,
            "delta_ci_upper": upper,
            "p_value": p,
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
