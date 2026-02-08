from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tushare_db import DataReader

from .config import Config
from .data.feature_store import FeatureStore
from .data.label import create_labeler
from .data.signal_loader import SignalLoader
from .models import create_model
from .split.splitters import create_splitter

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s}s"


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.reader = DataReader(db_path=config.db_path)

    def run(self) -> dict:
        if self.config.feature.mode == "image":
            return self._run_image()
        return self._run_tabular()

    def _run_tabular(self) -> dict:
        """表格特征 pipeline"""
        t_start = time.time()

        # 1. 加载信号
        signals = self._load_signals()

        # 2. 提取特征
        t = time.time()
        store = FeatureStore(self.config, self.reader)
        features = store.extract(signals)
        nan_ratio = features.isna().mean().mean()
        logger.info(
            f"[2/6] Features: {features.shape[1]} columns, "
            f"NaN ratio: {nan_ratio:.1%}, took {_fmt_duration(time.time() - t)}"
        )

        # 3. 计算标签
        t = time.time()
        labels = self._compute_labels(signals)
        logger.info(f"  Label computation took {_fmt_duration(time.time() - t)}")

        # 4. 去除标签缺失的样本
        valid_mask = labels.notna()
        signals = signals.loc[valid_mask].reset_index(drop=True)
        features = features.loc[valid_mask].reset_index(drop=True)
        labels = labels.loc[valid_mask].reset_index(drop=True)

        # NaN 诊断
        nan_per_col = features.isna().sum()
        nan_cols = nan_per_col[nan_per_col > 0].sort_values(ascending=False)
        total_nan = nan_per_col.sum()
        if len(nan_cols) > 0:
            logger.info(
                f"  NaN diagnosis: {total_nan} total NaNs across "
                f"{len(nan_cols)}/{features.shape[1]} columns"
            )
            for col, cnt in nan_cols.items():
                pct = cnt / len(features) * 100
                logger.info(f"    {col}: {cnt} ({pct:.1f}%)")
            features = features.fillna(0.0)
            logger.info(f"  Filled {total_nan} NaN values with 0")

        logger.info(f"[4/6] Valid samples: {len(labels)}")

        # 5. 划分
        splitter = create_splitter(self.config.split)
        split = splitter.split(features, labels, signals)
        self._log_split(split)

        # 6. 训练 + 评估
        t = time.time()
        model = create_model(self.config.model)
        model.fit(split.X_train, split.y_train, split.X_test, split.y_test)
        metrics = model.evaluate(split.X_test, split.y_test)
        logger.info(f"[6/6] Training + eval took {_fmt_duration(time.time() - t)}")

        importance = model.feature_importance()
        if importance is not None:
            logger.info(f"Top 10 features:\n{importance.head(10).to_string()}")

        self._finish(t_start)
        return metrics

    def _run_image(self) -> dict:
        """GAF 图像 pipeline (懒加载, 不全量载入内存)"""
        from .features.gaf import GAFStore, MultiScaleGAFStore

        t_start = time.time()
        multi_scale = self.config.feature.is_multi_scale

        # 1. 加载信号
        signals = self._load_signals()

        # 2. 生成 GAF 张量 (存为单个 .pt 文件)
        t = time.time()
        if multi_scale:
            gaf_store = MultiScaleGAFStore(self.config, self.reader)
        else:
            gaf_store = GAFStore(self.config, self.reader)
        path_dict = gaf_store.extract_paths(signals)
        logger.info(
            f"[2/6] GAF tensors: {len(path_dict)}/{len(signals)} available"
            f"{' (multi-scale)' if multi_scale else ''}, "
            f"took {_fmt_duration(time.time() - t)}"
        )

        # 3. 计算标签
        t = time.time()
        labels = self._compute_labels(signals)
        logger.info(f"  Label computation took {_fmt_duration(time.time() - t)}")

        # 4. 过滤: 只保留同时有 GAF 和标签的样本
        multi_target = isinstance(labels, pd.DataFrame)
        if multi_target:
            valid_indices = [
                idx for idx in range(len(signals))
                if idx in path_dict and labels.iloc[idx].notna().all()
            ]
        else:
            valid_indices = [
                idx for idx in range(len(signals))
                if idx in path_dict and pd.notna(labels.iloc[idx])
            ]
        logger.info(
            f"[4/6] Valid samples: {len(valid_indices)} "
            f"(dropped {len(signals) - len(valid_indices)})"
        )

        valid_signals = signals.iloc[valid_indices].reset_index(drop=True)
        valid_paths = [path_dict[i] for i in valid_indices]

        if multi_target:
            valid_labels = [labels.iloc[i].tolist() for i in valid_indices]
            vl_arr = np.array(valid_labels)
            for ci, col in enumerate(labels.columns):
                logger.info(
                    f"  {col}: mean={vl_arr[:, ci].mean():.4f} "
                    f"median={np.median(vl_arr[:, ci]):.4f} "
                    f"std={vl_arr[:, ci].std():.4f}"
                )
        else:
            valid_labels = [float(labels.iloc[i]) for i in valid_indices]
            vl_arr = np.array(valid_labels)
            logger.info(
                f"  Label stats: mean={vl_arr.mean():.4f} median={np.median(vl_arr):.4f} "
                f"std={vl_arr.std():.4f}"
            )

        # 5. 划分
        splitter = create_splitter(self.config.split)
        indices = list(range(len(valid_labels)))
        # splitter 需要 Series, 用第一列或唯一标签
        if multi_target:
            labels_series = pd.Series(vl_arr[:, 0])
        else:
            labels_series = pd.Series(valid_labels)

        dummy_features = pd.DataFrame({"_idx": indices})
        split = splitter.split(dummy_features, labels_series, valid_signals)

        train_idx = split.X_train["_idx"].tolist()
        test_idx = split.X_test["_idx"].tolist()

        X_train = [valid_paths[i] for i in train_idx]
        y_train = [valid_labels[i] for i in train_idx]
        X_test = [valid_paths[i] for i in test_idx]
        y_test = [valid_labels[i] for i in test_idx]

        logger.info(
            f"[5/6] Split: train={len(y_train)}, test={len(y_test)}"
        )

        # 6. 训练 + 评估
        t = time.time()
        model = create_model(self.config.model)
        model.fit(X_train, y_train, X_test, y_test)
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"[6/6] Training + eval took {_fmt_duration(time.time() - t)}")

        self._finish(t_start)
        return metrics

    # --- 共享方法 ---

    def _load_signals(self) -> pd.DataFrame:
        loader = SignalLoader(self.config.signal_file)
        signals = loader.load()
        n_stocks = signals["ts_code"].nunique()
        date_range = f"{signals['signal_date'].min()} ~ {signals['signal_date'].max()}"
        logger.info(
            f"[1/6] Loaded {len(signals)} signals, "
            f"{n_stocks} stocks, date range: {date_range}"
        )
        return signals

    def _compute_labels(self, signals: pd.DataFrame) -> pd.Series | pd.DataFrame:
        labels = self._load_or_compute_labels(signals)

        if isinstance(labels, pd.DataFrame):
            # 多目标 (e.g., RiskReturnLabeler)
            valid_mask = labels.notna().all(axis=1)
            n_valid = valid_mask.sum()
            n_missing = (~valid_mask).sum()
            logger.info(f"[3/6] Labels ({labels.columns.tolist()}): {n_valid} valid, {n_missing} missing")
            for col in labels.columns:
                vals = labels.loc[valid_mask, col]
                logger.info(
                    f"  {col}: mean={vals.mean():.4f} std={vals.std():.4f} "
                    f"median={vals.median():.4f} [{vals.min():.4f}, {vals.max():.4f}]"
                )
            return labels
        else:
            n_valid = labels.notna().sum()
            valid_labels = labels.dropna()
            logger.info(
                f"[3/6] Labels: {n_valid} valid, {labels.isna().sum()} missing | "
                f"mean={valid_labels.mean():.4f} std={valid_labels.std():.4f} "
                f"median={valid_labels.median():.4f} "
                f"[{valid_labels.min():.4f}, {valid_labels.max():.4f}]"
            )
            for t in [0.03, 0.05, 0.08, 0.10]:
                n_above = (valid_labels > t).sum()
                logger.info(f"  > {t:.0%}: {n_above} ({n_above / max(n_valid, 1):.1%})")
            return labels

    def _load_or_compute_labels(self, signals: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """带 parquet 缓存的标签计算"""
        cache_path = self._label_cache_path()

        if self.config.cache.enabled and cache_path.exists():
            cached = pd.read_parquet(cache_path)
            # 校验行数与 signals 一致
            if len(cached) == len(signals):
                logger.info(f"Loaded cached labels from {cache_path}")
                if cached.shape[1] == 1:
                    return cached.iloc[:, 0]
                return cached

            logger.info(
                f"Label cache size mismatch ({len(cached)} vs {len(signals)}), recomputing"
            )

        labeler = create_labeler(self.config.label)
        labels = labeler.compute(signals, self.reader)

        if self.config.cache.enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(labels, pd.DataFrame):
                labels.to_parquet(cache_path, index=False)
            else:
                labels.to_frame().to_parquet(cache_path, index=False)
            logger.info(f"Saved label cache to {cache_path}")

        return labels

    def _label_cache_path(self) -> Path:
        stem = self.config.signal_file_stem
        label_hash = self.config.label.config_hash()
        return Path(self.config.cache.dir) / "labels" / f"{stem}_{label_hash}.parquet"

    def _log_split(self, split):
        train_pos = (split.y_train == 1.0).sum()
        test_pos = (split.y_test == 1.0).sum()
        logger.info(
            f"[5/6] Split: train={len(split.y_train)} "
            f"({train_pos} pos, {len(split.y_train) - train_pos} neg), "
            f"test={len(split.y_test)} "
            f"({test_pos} pos, {len(split.y_test) - test_pos} neg)"
        )

    def _finish(self, t_start: float):
        total = time.time() - t_start
        logger.info(f"Pipeline finished in {_fmt_duration(total)}")
        self.reader.close()
