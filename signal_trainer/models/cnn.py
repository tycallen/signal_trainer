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

    def __init__(self, paths: list[str | Path], labels: list[float]):
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        tensor = torch.load(self.paths[idx], weights_only=True)
        return tensor, self.labels[idx]


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

    def __init__(self, in_channels: int = 10):
        super().__init__()
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
            nn.Linear(512, 1),
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
        return x.squeeze(1)


class CNNModel(BaseModel):
    def __init__(self, params: dict | None = None):
        defaults = {
            "in_channels": 10,
            "epochs": 30,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
        if params:
            defaults.update(params)
        self.params = defaults
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model: SimpleCNN | None = None

    def fit(
        self,
        X_train: list,
        y_train: list[float],
        X_val: list | None = None,
        y_val: list[float] | None = None,
    ) -> None:
        self.model = GAFResNet(in_channels=self.params["in_channels"]).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  GAFResNet: {n_params:,} parameters")

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
        logger.info(
            f"  Label stats: mean={y_arr.mean():.4f} std={y_arr.std():.4f} "
            f"median={np.median(y_arr):.4f} min={y_arr.min():.4f} max={y_arr.max():.4f}"
        )

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

        best_corr = -1.0
        best_state = None
        patience = self.params.get("patience", 15)
        no_improve = 0

        for epoch in range(self.params["epochs"]):
            self.model.train()
            total_loss = 0
            n_batches = 0
            for batch_x, batch_y in train_loader:
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
                corr = val_metrics["spearman"]
                val_msg = (
                    f" | val_mse={val_metrics['mse']:.6f} "
                    f"mae={val_metrics['mae']:.4f} "
                    f"corr={corr:.4f}"
                )
                # early stopping on val spearman
                if corr > best_corr:
                    best_corr = corr
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
                    f"best corr={best_corr:.4f}"
                )
                break

        # 恢复最优模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            logger.info(f"Restored best model (corr={best_corr:.4f})")

        logger.info(f"GAFResNet trained on {len(train_ds)} samples, device={self.device}")

    def predict(self, X) -> np.ndarray:
        """返回预测的连续值 (最大涨幅)"""
        self.model.eval()
        if isinstance(X, list):
            ds = GAFDataset(X, [0.0] * len(X))
            loader = DataLoader(ds, batch_size=self.params["batch_size"])
        else:
            loader = X

        all_preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
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

        mse = mean_squared_error(y_np, y_pred)
        mae = mean_absolute_error(y_np, y_pred)
        r2 = r2_score(y_np, y_pred)
        corr, p_value = spearmanr(y_np, y_pred)

        # 按阈值统计分类效果 (方便评估交易价值)
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
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "spearman": corr,
            "spearman_p": p_value,
        }

        logger.info(f"Evaluation: {metrics}")
        return metrics

    def feature_importance(self) -> pd.Series | None:
        return None

    def _eval_loader(self, loader: DataLoader, criterion) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x).cpu()
                all_preds.append(preds)
                all_labels.append(batch_y)

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        mse = mean_squared_error(labels, preds)
        mae = mean_absolute_error(labels, preds)
        corr, _ = spearmanr(labels, preds)

        return {"mse": mse, "mae": mae, "spearman": corr}
