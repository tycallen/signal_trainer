from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from .base import BaseModel

logger = logging.getLogger(__name__)


class GAFDataset(Dataset):
    """PyTorch Dataset: 从 .pt 文件路径懒加载 GAF 张量"""

    def __init__(self, paths: list[str | Path], labels: list[float] | list[list[float]]):
        self.paths = paths
        if isinstance(labels[0], (list, tuple, np.ndarray)):
            self.labels = torch.tensor(labels, dtype=torch.float32)  # (N, n_targets)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float32)  # (N,)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        tensor = torch.load(self.paths[idx], weights_only=True)
        return tensor, self.labels[idx]


class MultiScaleGAFDataset(Dataset):
    """PyTorch Dataset: 每个样本加载多个尺度的 GAF 张量"""

    def __init__(
        self,
        paths: list[list[str | Path]],
        labels: list[float] | list[list[float]],
    ):
        """
        paths: list of [path_30, path_60, path_120] per sample
        """
        self.paths = paths
        if isinstance(labels[0], (list, tuple, np.ndarray)):
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        tensors = [
            torch.load(p, weights_only=True) for p in self.paths[idx]
        ]
        return tensors, self.labels[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class GAFResNet(nn.Module):
    """ResNet-18 style, 适配 10 通道 GAF 输入"""

    def __init__(self, in_channels: int = 10, n_targets: int = 1):
        super().__init__()
        self.n_targets = n_targets
        # stem: 120x120 -> 60x60
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 60x60 -> 30x30
        )
        # residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)    # 30x30
        self.layer2 = self._make_layer(64, 128, 2, stride=2)   # 15x15
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 8x8
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, n_targets),
        )

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        if self.n_targets == 1:
            return x.squeeze(1)  # (batch,)
        return x  # (batch, n_targets)


class MultiScaleGAFResNet(nn.Module):
    """Shared ResNet encoder + feature fusion for multi-scale GAF"""

    def __init__(self, in_channels: int = 10, n_targets: int = 1, n_scales: int = 3):
        super().__init__()
        self.n_targets = n_targets
        self.n_scales = n_scales

        # Shared encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Regressor on concatenated features
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * n_scales, n_targets),
        )

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: input (B, C, H, H) -> output (B, 512)"""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, scales: list[torch.Tensor]) -> torch.Tensor:
        """
        scales: list of n_scales tensors, each (B, C, H_i, H_i)
        returns: (B, n_targets) or (B,)
        """
        features = [self._encode(x) for x in scales]
        fused = torch.cat(features, dim=1)  # (B, 512 * n_scales)
        out = self.regressor(fused)
        if self.n_targets == 1:
            return out.squeeze(1)
        return out


def _multi_scale_collate(batch):
    """Custom collate for MultiScaleGAFDataset: stack each scale separately"""
    scale_lists = [[] for _ in range(len(batch[0][0]))]
    labels = []
    for tensors, label in batch:
        for i, t in enumerate(tensors):
            scale_lists[i].append(t)
        labels.append(label)
    stacked_scales = [torch.stack(s) for s in scale_lists]
    return stacked_scales, torch.stack(labels)


class CNNModel(BaseModel):
    def __init__(self, params: dict | None = None):
        defaults = {
            "in_channels": 10,
            "n_targets": 1,
            "epochs": 30,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
        if params:
            defaults.update(params)
        self.params = defaults
        self.n_targets = self.params["n_targets"]
        self.n_scales = self.params.get("n_scales", 1)
        self.multi_scale = self.n_scales > 1
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model: GAFResNet | MultiScaleGAFResNet | None = None

    def fit(
        self,
        X_train: list,
        y_train: list[float] | list[list[float]],
        X_val: list | None = None,
        y_val: list[float] | list[list[float]] | None = None,
    ) -> None:
        if self.multi_scale:
            self.model = MultiScaleGAFResNet(
                in_channels=self.params["in_channels"],
                n_targets=self.n_targets,
                n_scales=self.n_scales,
            ).to(self.device)
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"  MultiScaleGAFResNet: {n_params:,} parameters, "
                f"{self.n_scales} scales, {self.n_targets} target(s)"
            )
        else:
            self.model = GAFResNet(
                in_channels=self.params["in_channels"],
                n_targets=self.n_targets,
            ).to(self.device)
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"  GAFResNet: {n_params:,} parameters, {self.n_targets} target(s)")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.params["epochs"], eta_min=1e-6,
        )
        criterion = nn.HuberLoss(delta=0.05)

        # 标签统计
        y_arr = np.array(y_train)
        if self.n_targets > 1:
            target_names = self.params.get("target_names", [f"target_{i}" for i in range(self.n_targets)])
            for i, name in enumerate(target_names):
                col = y_arr[:, i]
                logger.info(
                    f"  {name} stats: mean={col.mean():.4f} std={col.std():.4f} "
                    f"median={np.median(col):.4f} min={col.min():.4f} max={col.max():.4f}"
                )
        else:
            logger.info(
                f"  Label stats: mean={y_arr.mean():.4f} std={y_arr.std():.4f} "
                f"median={np.median(y_arr):.4f} min={y_arr.min():.4f} max={y_arr.max():.4f}"
            )

        if self.multi_scale:
            train_ds = MultiScaleGAFDataset(X_train, y_train)
            train_loader = DataLoader(
                train_ds,
                batch_size=self.params["batch_size"],
                shuffle=True,
                collate_fn=_multi_scale_collate,
            )
            val_loader = None
            if X_val is not None and y_val is not None:
                val_ds = MultiScaleGAFDataset(X_val, y_val)
                val_loader = DataLoader(
                    val_ds,
                    batch_size=self.params["batch_size"],
                    collate_fn=_multi_scale_collate,
                )
        else:
            train_ds = GAFDataset(X_train, y_train)
            train_loader = DataLoader(
                train_ds,
                batch_size=self.params["batch_size"],
                shuffle=True,
            )
            val_loader = None
            if X_val is not None and y_val is not None:
                val_ds = GAFDataset(X_val, y_val)
                val_loader = DataLoader(val_ds, batch_size=self.params["batch_size"])

        best_score = -1.0
        best_state = None
        patience = self.params.get("patience", 15)
        no_improve = 0

        for epoch in range(self.params["epochs"]):
            self.model.train()
            total_loss = 0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                if self.multi_scale:
                    batch_x = [t.to(self.device) for t in batch_x]
                else:
                    batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            val_msg = ""
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_metrics = self._eval_loader(val_loader, criterion)
                score = val_metrics["score"]
                val_msg = f" | val_loss={val_metrics['loss']:.6f} score={score:.4f}"
                if self.n_targets > 1:
                    val_msg += f" (ret_corr={val_metrics.get('spearman_0', 0):.4f})"
                # early stopping
                if score > best_score:
                    best_score = score
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                    val_msg += " *"
                else:
                    no_improve += 5  # 每 5 epoch 检查一次

            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Epoch {epoch+1}/{self.params['epochs']}: "
                    f"train_loss={avg_loss:.6f} lr={lr:.2e}{val_msg}"
                )

            if no_improve >= patience:
                logger.info(
                    f"  Early stopping at epoch {epoch+1}, "
                    f"best score={best_score:.4f}"
                )
                break

        # 恢复最优模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            logger.info(f"Restored best model (score={best_score:.4f})")

        logger.info(f"GAFResNet trained on {len(train_ds)} samples, device={self.device}")

    def predict(self, X) -> np.ndarray:
        """返回预测值: 单目标 (N,), 双目标 (N, 2)"""
        self.model.eval()
        if isinstance(X, list):
            dummy_labels = [[0.0] * self.n_targets for _ in range(len(X))] if self.n_targets > 1 else [0.0] * len(X)
            if self.multi_scale:
                ds = MultiScaleGAFDataset(X, dummy_labels)
                loader = DataLoader(
                    ds, batch_size=self.params["batch_size"],
                    collate_fn=_multi_scale_collate,
                )
            else:
                ds = GAFDataset(X, dummy_labels)
                loader = DataLoader(ds, batch_size=self.params["batch_size"])
        else:
            loader = X

        all_preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                if self.multi_scale:
                    batch_x = [t.to(self.device) for t in batch_x]
                else:
                    batch_x = batch_x.to(self.device)
                preds = self.model(batch_x).cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds)

    def predict_proba(self, X) -> np.ndarray:
        """回归模型无概率输出, 返回预测值"""
        return self.predict(X)

    def evaluate(self, X, y) -> dict[str, float]:
        y_np = np.array(y)
        y_pred = self.predict(X)

        if self.n_targets > 1:
            return self._evaluate_multi(y_np, y_pred)
        return self._evaluate_single(y_np, y_pred)

    def _evaluate_single(self, y_np: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        mse = mean_squared_error(y_np, y_pred)
        mae = mean_absolute_error(y_np, y_pred)
        r2 = r2_score(y_np, y_pred)
        corr, p_value = spearmanr(y_np, y_pred)

        for threshold in [0.03, 0.05, 0.08]:
            y_cls = (y_np > threshold).astype(int)
            pred_cls = (y_pred > threshold).astype(int)
            n_pred_pos = pred_cls.sum()
            if n_pred_pos > 0:
                precision = (y_cls[pred_cls == 1] == 1).mean()
                recall = (pred_cls[y_cls == 1] == 1).mean() if (y_cls == 1).sum() > 0 else 0
            else:
                precision = 0
                recall = 0
            logger.info(
                f"  Threshold {threshold:.0%}: "
                f"pred_pos={n_pred_pos}, precision={precision:.3f}, recall={recall:.3f}"
            )

        metrics = {
            "mse": mse, "mae": mae, "r2": r2,
            "spearman": corr, "spearman_p": p_value,
        }
        logger.info(f"Evaluation: {metrics}")
        return metrics

    def _evaluate_multi(self, y_np: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        target_names = self.params.get("target_names", ["max_return", "max_drawdown"])
        metrics = {}

        for i, name in enumerate(target_names):
            yi = y_np[:, i]
            pi = y_pred[:, i]
            mse = mean_squared_error(yi, pi)
            mae = mean_absolute_error(yi, pi)
            r2 = r2_score(yi, pi)
            corr, _ = spearmanr(yi, pi)
            metrics[f"{name}_mse"] = mse
            metrics[f"{name}_mae"] = mae
            metrics[f"{name}_r2"] = r2
            metrics[f"{name}_spearman"] = corr
            logger.info(
                f"  {name}: mse={mse:.6f} mae={mae:.4f} r2={r2:.4f} spearman={corr:.4f}"
            )

        # 风险调整过滤: 预测涨幅 > 阈值 且 回撤 > 阈值
        pred_ret = y_pred[:, 0]
        pred_dd = y_pred[:, 1]
        true_ret = y_np[:, 0]
        true_dd = y_np[:, 1]

        for ret_thr, dd_thr in [(0.05, -0.03), (0.08, -0.03), (0.05, -0.05)]:
            mask = (pred_ret > ret_thr) & (pred_dd > dd_thr)
            n_selected = mask.sum()
            if n_selected > 0:
                avg_ret = true_ret[mask].mean()
                avg_dd = true_dd[mask].mean()
                # 实际也满足条件的比例
                actual_good = ((true_ret[mask] > ret_thr) & (true_dd[mask] > dd_thr)).mean()
                logger.info(
                    f"  Filter ret>{ret_thr:.0%} & dd>{dd_thr:.0%}: "
                    f"n={n_selected}, avg_ret={avg_ret:.4f}, avg_dd={avg_dd:.4f}, "
                    f"precision={actual_good:.3f}"
                )
                metrics[f"filter_{ret_thr}_{dd_thr}_n"] = float(n_selected)
                metrics[f"filter_{ret_thr}_{dd_thr}_precision"] = float(actual_good)
            else:
                logger.info(f"  Filter ret>{ret_thr:.0%} & dd>{dd_thr:.0%}: n=0")

        logger.info(f"Evaluation summary: {metrics}")
        return metrics

    def feature_importance(self) -> pd.Series | None:
        return None

    def _eval_loader(self, loader: DataLoader, criterion) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                if self.multi_scale:
                    batch_x = [t.to(self.device) for t in batch_x]
                else:
                    batch_x = batch_x.to(self.device)
                preds = self.model(batch_x).cpu()
                all_preds.append(preds)
                all_labels.append(batch_y)

        preds_t = torch.cat(all_preds)
        labels_t = torch.cat(all_labels)
        loss = criterion(preds_t, labels_t).item()

        preds = preds_t.numpy()
        labels = labels_t.numpy()

        if self.n_targets > 1:
            # 用第一个目标 (max_return) 的 spearman 作为 early stopping score
            corr0, _ = spearmanr(labels[:, 0], preds[:, 0])
            corr1, _ = spearmanr(labels[:, 1], preds[:, 1])
            return {
                "loss": loss,
                "spearman_0": corr0,
                "spearman_1": corr1,
                "score": (corr0 + corr1) / 2,
            }

        corr, _ = spearmanr(labels, preds)
        return {"loss": loss, "spearman": corr, "score": corr}
