######################################
# 2026-2-2 #
# Model Trainer Adapter #
######################################
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from benchmarks.metrics import compute_metrics
from core.utils import prepare_labels


class BaseAdapter:
    def __init__(self, spec, device):
        self.spec = spec
        self.device = device

    def model_input(self, batch):
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (tuple, list)):
            # If batch is (sequence_dict, graph_list), keep tuple for models like TRANS
            if len(batch) == 2 and isinstance(batch[0], dict) and not isinstance(batch[1], dict):
                return batch
            return batch[0]
        return batch

    def forward(self, model, batch):
        raise NotImplementedError

    def parse_output(self, output):
        # Support pyhealth outputs: dict with "logit"/"logits" and optional "loss"/"y_prob"
        if isinstance(output, dict):
            logits = None
            if "logits" in output:
                logits = output["logits"]
            elif "logit" in output:
                logits = output["logit"]
            y_prob = output.get("y_prob")
            loss = output.get("loss")
            return logits, loss, y_prob
        if isinstance(output, tuple):
            return output[0], None, None
        return output, None, None

    def loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

    def labels_from_batch(self, batch):
        label_key = self.spec.label_key
        if isinstance(batch, dict):
            raw = batch[label_key]
        else:
            raw = batch[0][label_key]
        labels = prepare_labels(raw, self.spec.label_tokenizer).to(self.device)
        return labels


class DiagAdapter(BaseAdapter):
    def forward(self, model, batch):
        # diag models return logits or (logits, ...)
        data = self.model_input(batch)
        if isinstance(data, dict):
            try:
                out = model(**data)
            except TypeError:
                out = model(data)
        else:
            out = model(data)
        return self.parse_output(out)


class DrugAdapter(BaseAdapter):
    def forward(self, model, batch):
        # pyhealth models typically return dict or logits
        data = self.model_input(batch)
        if isinstance(data, dict):
            try:
                out = model(**data)
            except TypeError:
                out = model(data)
        else:
            out = model(data)
        return self.parse_output(out)


def pick_adapter(args, spec, device):
    if args.task == "drug":
        return DrugAdapter(spec, device)
    return DiagAdapter(spec, device)


class SimpleTrainer:
    def __init__(self, args, spec, device):
        self.args = args
        self.spec = spec
        self.device = device
        self.adapter = pick_adapter(args, spec, device)

    def _prepare_model(self, model):
        model = model.to(self.device)
        if hasattr(model, "device"):
            try:
                model.device = self.device
            except Exception:
                pass
        return model

    def fit(self, model):
        model = self._prepare_model(model)
        train_loader, val_loader, _ = self.spec.dataloaders
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = None
        if getattr(self.args, "scheduler", "none") == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=getattr(self.args, "t_max", self.args.epochs),
                eta_min=getattr(self.args, "eta_min", 1e-6),
            )
        os.makedirs(self.args.output_dir, exist_ok=True)
        ckpt_path = self._ckpt_path()

        best_val = float("inf")
        epoch_iter = range(1, self.args.epochs + 1)
        epoch_pbar = None
        if self.args.show_progress:
            epoch_pbar = tqdm(epoch_iter, desc="Epoch", leave=True)
            epoch_iter = epoch_pbar
        for epoch in epoch_iter:
            train_loss, _ = self._run_epoch(
                train_loader, model, optimizer, train=True, do_metrics=False
            )
            val_loss, val_metrics = self._run_epoch(
                val_loader, model, optimizer=None, train=False, do_metrics=True
            )
            if epoch_pbar is not None:
                epoch_pbar.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )
            if scheduler is not None:
                scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)

            if self.args.eval_every > 0 and epoch % self.args.eval_every == 0:
                # Only print test metrics periodically
                _, _, test_loader = self.spec.dataloaders
                _, test_metrics = self._run_epoch(
                    test_loader, model, optimizer=None, train=False, do_metrics=True, compute_loss=False
                )
                self._print_test_epoch(epoch, test_metrics)

        return ckpt_path

    def evaluate(self, model, ckpt_path: Optional[str] = None):
        _, _, test_loader = self.spec.dataloaders
        if ckpt_path is None:
            ckpt_path = self._ckpt_path()
        if ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        model = self._prepare_model(model)
        _, test_metrics = self._run_epoch(
            test_loader, model, optimizer=None, train=False, do_metrics=True, compute_loss=False
        )
        return None, test_metrics

    def _run_epoch(
        self,
        data_loader,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        train: bool,
        do_metrics: bool,
        compute_loss: bool = True,
    ) -> Tuple[float, Optional[dict]]:
        loss_sum = 0.0
        y_true_all = []
        y_prob_all = []

        model.train() if train else model.eval()
        for batch in tqdm(data_loader, disable=True):
            if train:
                optimizer.zero_grad()

            logits, model_loss, y_prob = self.adapter.forward(model, batch)
            labels = self.adapter.labels_from_batch(batch)
            loss = None
            if compute_loss:
                if model_loss is not None:
                    loss = model_loss
                else:
                    if logits is None:
                        raise ValueError("Model did not return logits or loss; cannot compute loss.")
                    loss = self.adapter.loss(logits, labels)

            if train:
                if loss is None:
                    raise ValueError("Loss is required for training but was not computed.")
                loss.backward()
                optimizer.step()

            if compute_loss and loss is not None:
                loss_sum += float(loss.detach().cpu().item())

            if do_metrics:
                y_true_all.append(labels.detach().cpu().numpy())
                if y_prob is not None:
                    y_prob_all.append(y_prob.detach().cpu().numpy())
                else:
                    if logits is None:
                        raise ValueError("Model did not return logits or y_prob; cannot compute metrics.")
                    y_prob_all.append(torch.sigmoid(logits).detach().cpu().numpy())

        avg_loss = loss_sum / max(1, len(data_loader)) if compute_loss else 0.0
        metrics = None
        if do_metrics and y_true_all:
            y_true = np.concatenate(y_true_all, axis=0)
            y_prob = np.concatenate(y_prob_all, axis=0)
            metrics = compute_metrics(
                y_true=y_true,
                y_prob=y_prob,
                metrics=[
                    "f1_samples",
                    "jaccard_samples",
                    "pr_auc_samples",
                    "precision_samples",
                    "roc_auc_samples",
                    "code_level",
                    "visit_level",
                ],
            )
        return avg_loss, metrics

    def _ckpt_path(self):
        return os.path.join(
            self.args.output_dir,
            f"trained_{self.args.task}_{self.args.model}_{self.args.dataset}.ckpt",
        )

    def _print_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: dict):
        msg = (
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )
        print(msg)
        if not metrics:
            return
        if "f1_samples" in metrics:
            print(
                f"[metrics] F1={metrics['f1_samples']:.4f} "
                f"Jaccard={metrics['jaccard_samples']:.4f} "
                f"PR-AUC={metrics['pr_auc_samples']:.4f} "
                f"Precision={metrics['precision_samples']:.4f} "
                f"ROC-AUC={metrics['roc_auc_samples']:.4f}"
            )
        if "visit_level" in metrics:
            print(f"[metrics] visit_level={metrics['visit_level']}")
        if "code_level" in metrics:
            print(f"[metrics] code_level={metrics['code_level']}")

    def _print_test_epoch(self, epoch: int, metrics: dict):
        print(f"[epoch {epoch}] test metrics")
        if not metrics:
            return
        if "f1_samples" in metrics:
            print(
                f"[metrics] F1={metrics['f1_samples']:.4f} "
                f"Jaccard={metrics['jaccard_samples']:.4f} "
                f"PR-AUC={metrics['pr_auc_samples']:.4f} "
                f"Precision={metrics['precision_samples']:.4f} "
                f"ROC-AUC={metrics['roc_auc_samples']:.4f}"
            )
        if "visit_level" in metrics:
            print(f"[metrics] visit_level={metrics['visit_level']}")
        if "code_level" in metrics:
            print(f"[metrics] code_level={metrics['code_level']}")


def pick_trainer(args, spec, device):
    return SimpleTrainer(args, spec, device)
