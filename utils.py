import random

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BalancedAccuracy(torchmetrics.Metric):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state(
            "correct_per_class",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_per_class",
            default=torch.zeros(num_classes),
            dist_reduce_fx="sum",
        )

    def update(self, preds, target):
        if preds.dim() > 1:
            preds = preds.argmax(dim=-1)
        for c in range(self.num_classes):
            class_mask = (target == c)
            self.total_per_class[c] += class_mask.sum()
            self.correct_per_class[c] += ((preds == c) & class_mask).sum()

    def compute(self):
        return (self.correct_per_class / self.total_per_class).mean()


def build_criterion(
    task,
    df_train,
    num_classes,
    device,
):
    if task == "binary":
        n_pos = int((df_train["label"] == 1).sum())
        n_neg = int((df_train["label"] == 0).sum())

        pos_weight_value = n_neg / n_pos

        pos_weight = torch.tensor([pos_weight_value], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return criterion

    elif task == "multiclass":
        counts = df_train["label"].value_counts().sort_index()
        freq = counts.to_numpy().astype(float)
        class_weights = 1.0 / freq
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.tensor(
            class_weights, dtype=torch.float32, device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        return criterion


def build_metrics(
    task,
    num_classes,
):
    if task == "binary":
        metrics = {
            "1.AUC": torchmetrics.AUROC(task="binary"),
        }
        return metrics, "1.AUC"

    elif task == "multiclass":
        metrics = {
            "1.BalancedAcc": BalancedAccuracy(num_classes=num_classes),
        }
        return metrics, "1.BalancedAcc"


def _compute_youden_threshold(
    probs, labels,
):
    fpr, tpr, thr = roc_curve(labels, probs)
    if len(thr) > 1:
        fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]
    j = tpr - fpr  # = sens + spec - 1
    best_idx = int(np.argmax(j))
    return float(thr[best_idx]), float(j[best_idx])


def _binary_cm_at_threshold(
    probs, labels, thr,
):
    y_pred = (probs >= thr).astype(int)
    tp = int(((y_pred == 1) & (labels == 1)).sum())
    fn = int(((y_pred == 0) & (labels == 1)).sum())
    tn = int(((y_pred == 0) & (labels == 0)).sum())
    fp = int(((y_pred == 1) & (labels == 0)).sum())
    return {
        "threshold": float(thr),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


@torch.inference_mode()
def _collect_predictions(
    model,
    loader,
    device,
    task,
):
    model.eval()
    all_preds = []
    all_labels = []
    loss_accum, n = 0.0, 0

    criterion = nn.BCEWithLogitsLoss(reduction="sum") if task == "binary" else None

    for batch in loader:
        patch_features = batch["patch_features"].to(device, non_blocking=True)
        type_ids = batch["type_ids"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).view(-1)

        out = model(patch_features, type_ids, padding_mask)
        logits = out["logits"]

        if task == "binary":
            logits_flat = logits.squeeze(-1)
            probs = torch.sigmoid(logits_flat).detach().cpu().numpy()
            all_preds.append(probs)
            loss_accum += criterion(logits_flat, labels.float()).item()
            n += labels.numel()
        elif task == "multiclass":
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.append(probs)

        all_labels.append(labels.detach().cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    loss = loss_accum / n if n > 0 else 0.0
    return preds, labels, loss


def test_with_bootstrap(
    model,
    loader,
    device,
    task,
    n_bootstrap,
    seed=42,
    threshold_tuning_loader=None,
):
    preds, labels, loss = _collect_predictions(model, loader, device, task)
    n_samples = len(labels)

    rng = np.random.RandomState(seed)

    if task == "binary":
        val_preds, val_labels, _ = _collect_predictions(
            model, threshold_tuning_loader, device, task,
        )
        youden_thr, _ = _compute_youden_threshold(val_preds, val_labels)

        point = {
            "auroc": roc_auc_score(labels, preds),
            "auprc": average_precision_score(labels, preds),
        }
        pred_bin = (preds >= 0.5).astype(int)
        tp = int(((pred_bin == 1) & (labels == 1)).sum())
        fn = int(((pred_bin == 0) & (labels == 1)).sum())
        tn = int(((pred_bin == 0) & (labels == 0)).sum())
        fp = int(((pred_bin == 1) & (labels == 0)).sum())
        point["sensitivity"] = tp / (tp + fn)
        point["specificity"] = tn / (tn + fp)

        youden_cm = _binary_cm_at_threshold(preds, labels, youden_thr)
        point["youden_threshold"] = youden_cm["threshold"]
        point["youden_sensitivity"] = youden_cm["sensitivity"]
        point["youden_specificity"] = youden_cm["specificity"]
        point["youden_tp"] = youden_cm["tp"]
        point["youden_fn"] = youden_cm["fn"]
        point["youden_tn"] = youden_cm["tn"]
        point["youden_fp"] = youden_cm["fp"]

        aurocs, auprcs = [], []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n_samples, size=n_samples)
            aurocs.append(roc_auc_score(labels[idx], preds[idx]))
            auprcs.append(average_precision_score(labels[idx], preds[idx]))

        result = {
            **point,
            "auroc_ci_lower": float(np.percentile(aurocs, 2.5)),
            "auroc_ci_upper": float(np.percentile(aurocs, 97.5)),
            "auprc_ci_lower": float(np.percentile(auprcs, 2.5)),
            "auprc_ci_upper": float(np.percentile(auprcs, 97.5)),
            "loss": loss,
            "n_samples": n_samples,
            "n_pos": int(labels.sum()),
            "n_bootstrap": n_bootstrap,
        }
        return result

    else:
        from sklearn.metrics import balanced_accuracy_score
        pred_cls = preds.argmax(axis=-1)
        point = {
            "AUC_OVR_macro": roc_auc_score(
                labels, preds, multi_class="ovr", average="macro",
            ),
            "Accuracy_balanced": balanced_accuracy_score(labels, pred_cls),
        }
        aurocs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n_samples, size=n_samples)
            aurocs.append(
                roc_auc_score(
                    labels[idx], preds[idx],
                    multi_class="ovr", average="macro",
                )
            )

        return {
            **point,
            "auroc_ci_lower": float(np.percentile(aurocs, 2.5)),
            "auroc_ci_upper": float(np.percentile(aurocs, 97.5)),
            "loss": loss,
            "n_samples": n_samples,
            "n_pos": int((labels == 1).sum()),
            "n_bootstrap": n_bootstrap,
        }


def format_test_json(
    result,
):
    def r4(v):
        return round(v, 4) if isinstance(v, float) else v

    out = {}
    for k, v in result.items():
        out[k] = r4(v)
    return out
