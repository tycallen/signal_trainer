from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tushare_db import DataReader

from ..config import Config

logger = logging.getLogger(__name__)

# OHLCV 字段及其对应的通道索引:
# open:  GASF(0), GADF(1)
# close: GASF(2), GADF(3)
# high:  GASF(4), GADF(5)
# low:   GASF(6), GADF(7)
# vol:   GASF(8), GADF(9)
SERIES_FIELDS = ["open", "close", "high", "low", "vol"]
N_CHANNELS = len(SERIES_FIELDS) * 2  # 10


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    """归一化到 [0, 1], 然后映射到 [-1, 1] 用于 arccos"""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-10:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn) * 2 - 1


def _gasf(cos_values: np.ndarray) -> np.ndarray:
    """Gramian Angular Summation Field"""
    sin_values = np.sqrt(np.clip(1 - cos_values ** 2, 0, 1))
    # GASF[i,j] = cos(phi_i + phi_j) = cos_i * cos_j - sin_i * sin_j
    return np.outer(cos_values, cos_values) - np.outer(sin_values, sin_values)


def _gadf(cos_values: np.ndarray) -> np.ndarray:
    """Gramian Angular Difference Field"""
    sin_values = np.sqrt(np.clip(1 - cos_values ** 2, 0, 1))
    # GADF[i,j] = sin(phi_i - phi_j) = sin_i * cos_j - cos_i * sin_j
    return np.outer(sin_values, cos_values) - np.outer(cos_values, sin_values)


def series_to_gaf(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将一维时序转为 GASF + GADF 矩阵 (NxN each)"""
    cos_values = _normalize_minmax(series)
    return _gasf(cos_values), _gadf(cos_values)


def ohlcv_to_gaf_tensor(daily_df: pd.DataFrame, lookback: int) -> torch.Tensor | None:
    """
    将 OHLCV DataFrame 转为 10 通道 GAF 张量.
    返回 shape: (10, lookback, lookback) 的 float32 tensor.
    如果数据不足返回 None.
    """
    if len(daily_df) < lookback:
        return None

    # 取最后 lookback 天
    window = daily_df.tail(lookback)
    channels = []

    for field in SERIES_FIELDS:
        values = window[field].values.astype(np.float64)
        gasf, gadf = series_to_gaf(values)
        channels.append(gasf)
        channels.append(gadf)

    # (10, lookback, lookback)
    tensor = np.stack(channels, axis=0).astype(np.float32)
    return torch.from_numpy(tensor)


class GAFStore:
    """批量生成 GAF 张量, 按单个信号缓存为 .pt 文件"""

    def __init__(self, config: Config, reader: DataReader):
        self.config = config
        self.reader = reader
        self.lookback = config.feature.lookback
        self.adj = config.feature.adj
        self.cache_dir = self._get_cache_dir()

    def extract(self, signals: pd.DataFrame) -> dict[int, torch.Tensor]:
        """
        返回 {signal_index: tensor(10, L, L)} 字典.
        无法生成的信号不包含在返回值中.
        """
        all_keys = list(zip(signals["ts_code"], signals["signal_date"]))

        # 检查哪些已缓存
        cached_count = 0
        missing_keys = set()
        for key in all_keys:
            if self.config.cache.enabled and self._tensor_path(key).exists():
                cached_count += 1
            else:
                missing_keys.add(key)

        if cached_count > 0:
            logger.info(f"Found {cached_count} cached GAF tensors")

        if missing_keys:
            logger.info(
                f"Computing GAF for {len(missing_keys)} signals "
                f"({cached_count} cached, {len(all_keys)} total)"
            )
            self._compute_and_save(signals, missing_keys)

        # 加载所有张量
        result = {}
        for idx, key in enumerate(all_keys):
            pt_path = self._tensor_path(key)
            if pt_path.exists():
                result[idx] = torch.load(pt_path, weights_only=True)

        logger.info(f"Loaded {len(result)}/{len(all_keys)} GAF tensors")
        return result

    def extract_paths(self, signals: pd.DataFrame) -> dict[int, Path]:
        """
        确保所有 GAF 已生成, 返回 {signal_index: .pt 文件路径} 字典.
        不加载张量到内存, 用于懒加载.
        """
        all_keys = list(zip(signals["ts_code"], signals["signal_date"]))

        cached_count = 0
        missing_keys = set()
        for key in all_keys:
            if self.config.cache.enabled and self._tensor_path(key).exists():
                cached_count += 1
            else:
                missing_keys.add(key)

        if cached_count > 0:
            logger.info(f"Found {cached_count} cached GAF tensors")

        if missing_keys:
            logger.info(
                f"Computing GAF for {len(missing_keys)} signals "
                f"({cached_count} cached, {len(all_keys)} total)"
            )
            self._compute_and_save(signals, missing_keys)

        # 返回路径映射
        result = {}
        for idx, key in enumerate(all_keys):
            pt_path = self._tensor_path(key)
            if pt_path.exists():
                result[idx] = pt_path

        logger.info(f"Available {len(result)}/{len(all_keys)} GAF tensors")
        return result

    def _compute_and_save(
        self,
        signals: pd.DataFrame,
        missing_keys: set[tuple[str, str]],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        missing_df = signals[
            signals.apply(
                lambda r: (r["ts_code"], r["signal_date"]) in missing_keys, axis=1
            )
        ]

        stock_groups = list(missing_df.groupby("ts_code"))
        total_stocks = len(stock_groups)
        saved_count = 0

        for i, (ts_code, group) in enumerate(stock_groups, 1):
            if i % 200 == 0 or i == total_stocks:
                logger.info(
                    f"  GAF progress: {i}/{total_stocks} stocks, "
                    f"{saved_count} tensors saved"
                )

            max_date = group["signal_date"].max()
            daily = self.reader.get_stock_daily(
                ts_code, start_date="19900101", end_date=max_date, adj=self.adj,
            )
            if daily.empty:
                continue

            daily = daily.sort_values("trade_date").reset_index(drop=True)
            date_list = daily["trade_date"].tolist()

            for _, row in group.iterrows():
                signal_date = row["signal_date"]
                valid_dates = [d for d in date_list if d <= signal_date]
                if len(valid_dates) < self.lookback:
                    continue
                end_idx = date_list.index(valid_dates[-1])
                start_idx = end_idx - self.lookback + 1
                if start_idx < 0:
                    continue
                window = daily.iloc[start_idx:end_idx + 1]

                tensor = ohlcv_to_gaf_tensor(window, self.lookback)
                if tensor is not None:
                    key = (ts_code, signal_date)
                    if self.config.cache.enabled:
                        torch.save(tensor, self._tensor_path(key))
                    saved_count += 1

    def _tensor_path(self, key: tuple[str, str]) -> Path:
        ts_code, signal_date = key
        # 000001.SZ_20240115.pt
        safe_code = ts_code.replace(".", "_")
        return self.cache_dir / f"{safe_code}_{signal_date}.pt"

    def _get_cache_dir(self) -> Path:
        import hashlib
        key = f"{self.config.signal_file_stem}_gaf_{self.lookback}_{self.adj}"
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        return Path(self.config.cache.dir) / "gaf" / f"{self.config.signal_file_stem}_{h}"
