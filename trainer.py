import json
import os

import torch
from torch.amp import autocast, GradScaler

from utils import (
    test_with_bootstrap,
    format_test_json,
)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        test_h_loader,
        criterion,
        metrics,
        optimizer,
        scheduler,
        device,
        task,
        primary_key,
        log_dir,
        patience,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_h_loader = test_h_loader
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.task = task
        self.primary_key = primary_key
        self.log_dir = log_dir
        self.patience = patience

        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        self.scaler = GradScaler("cuda")

        self.best_primary = float("-inf")
        self.patience_counter = 0

    def _prepare_logits_and_preds(
        self, logits,
    ):
        if self.task == "binary":
            logits_for_loss = logits.squeeze(-1)
        elif self.task == "multiclass":
            logits_for_loss = logits

        with torch.no_grad():
            if self.task == "binary":
                preds = torch.sigmoid(logits.squeeze(-1))
            elif self.task == "multiclass":
                preds = torch.softmax(logits, dim=-1)

        return logits_for_loss, preds

    def _update_metrics(
        self,
        preds_all,
        labels_all,
    ):
        out = {}
        for name, metric in self.metrics.items():
            metric.reset()
            metric.update(preds_all, labels_all.long())
            value = metric.compute()
            out[name] = value.detach().item() if torch.is_tensor(value) else float(value)
        return out

    def _run_epoch(self, is_train):
        self.model.train() if is_train else self.model.eval()
        loader = self.train_loader if is_train else self.val_loader

        for m in self.metrics.values():
            m.to(self.device)

        preds_all = []
        labels_all = []

        for batch in loader:
            patch_features = batch["patch_features"].to(self.device, non_blocking=True)
            type_ids = batch["type_ids"].to(self.device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True).view(-1)

            with torch.set_grad_enabled(is_train):
                with autocast("cuda"):
                    outputs = self.model(patch_features, type_ids, padding_mask)
                    logits = outputs["logits"]
                    logits_for_loss, preds = self._prepare_logits_and_preds(logits)
                    loss = self.criterion(logits_for_loss, labels)

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds_all.append(preds.cpu())
                labels_all.append(labels.cpu())

        if is_train:
            self.scheduler.step()
            return None

        preds_cat = torch.cat(preds_all, dim=0).to(self.device)
        labels_cat = torch.cat(labels_all, dim=0).to(self.device)
        return self._update_metrics(preds_cat, labels_cat)

    def train_epoch(self):
        return self._run_epoch(is_train=True)

    @torch.inference_mode()
    def valid_epoch(self):
        return self._run_epoch(is_train=False)

    def fit(self, epochs):
        for _ in range(epochs):
            self.train_epoch()
            valid_out = self.valid_epoch()

            current_primary = valid_out[self.primary_key]
            better = current_primary > self.best_primary

            if better:
                self.best_primary = current_primary
                save_path = os.path.join(self.log_dir, "checkpoints", "best_model.pth")
                self._save_best(save_path)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                break

    def _save_best(self, path):
        torch.save(self.model.state_dict_for_save(), path)

    def run_bootstrap_test(
        self,
        n_bootstrap,
        seed=42,
    ):
        eval_loaders = [("test", self.test_loader), ("test_h", self.test_h_loader)]

        best_ckpt_path = os.path.join(self.log_dir, "checkpoints", "best_model.pth")

        state = torch.load(best_ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict_for_resume(state)

        combined_json = {"checkpoint_path": best_ckpt_path}
        for name, loader in eval_loaders:
            result = test_with_bootstrap(
                model=self.model,
                loader=loader,
                device=self.device,
                task=self.task,
                n_bootstrap=n_bootstrap,
                seed=seed,
                threshold_tuning_loader=self.val_loader if self.task == "binary" else None,
            )
            combined_json[name] = format_test_json(result)

        json_path = os.path.join(self.log_dir, "test_results.json")
        with open(json_path, "w") as f:
            json.dump(combined_json, f, indent=2, ensure_ascii=False)
