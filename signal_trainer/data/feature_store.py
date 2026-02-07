from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tushare_db import DataReader

from ..config import Config
from ..features.base import BaseExtractor
from ..features.money_flow import MoneyFlowExtractor
from ..features.technical import TechnicalExtractor

logger = logging.getLogger(__name__)

EXTRACTOR_REGISTRY: dict[str, type] = {
    "technical": TechnicalExtractor,
    "money_flow": MoneyFlowExtractor,
}


class FeatureStore:
    """从 DuckDB 提取特征, 带 parquet 缓存"""

    def __init__(self, config: Config, reader: DataReader):
        self.config = config
        self.reader = reader
        self.extractors = self._build_extractors()
        self.cache_dir = Path(config.cache.dir) / "features"

    def _build_extractors(self) -> list[BaseExtractor]:
        extractors = []
        for name in self.config.feature.extractors:
            cls = EXTRACTOR_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown extractor: {name}")
            # MoneyFlowExtractor 需要 reader
            if cls is MoneyFlowExtractor:
                extractors.append(cls(reader=self.reader))
            else:
                extractors.append(cls())
        return extractors

    def extract(self, signals: pd.DataFrame) -> pd.DataFrame:
        """提取所有信号的特征, 返回 DataFrame (index 对齐 signals)"""
        cache_path = self._cache_path()

        # 尝试加载缓存
        cached_df = None
        cached_keys: set[tuple[str, str]] = set()
        if self.config.cache.enabled and cache_path.exists():
            cached_df = pd.read_parquet(cache_path)
            cached_keys = set(
                zip(cached_df["ts_code"], cached_df["signal_date"])
            )
            logger.info(f"Loaded {len(cached_keys)} cached features")

        # 找出需要计算的信号
        all_keys = set(zip(signals["ts_code"], signals["signal_date"]))
        missing_keys = all_keys - cached_keys

        if missing_keys:
            logger.info(
                f"Computing features for {len(missing_keys)} signals "
                f"({len(cached_keys)} cached, {len(all_keys)} total)"
            )
            new_rows = self._compute_batch(signals, missing_keys)
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                if cached_df is not None:
                    cached_df = pd.concat([cached_df, new_df], ignore_index=True)
                else:
                    cached_df = new_df
                # 保存缓存
                if self.config.cache.enabled:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cached_df.to_parquet(cache_path, index=False)
                    logger.info(f"Saved cache to {cache_path}")

        if cached_df is None or cached_df.empty:
            return pd.DataFrame()

        # 按 signals 的顺序对齐返回
        result = signals.merge(cached_df, on=["ts_code", "signal_date"], how="left")
        # 去掉 ts_code, signal_date 列, 只保留特征
        feature_cols = [c for c in result.columns if c not in ("ts_code", "signal_date")]
        return result[feature_cols]

    def _compute_batch(
        self, signals: pd.DataFrame, missing_keys: set[tuple[str, str]]
    ) -> list[dict]:
        rows = []
        lookback = self.config.feature.lookback
        adj = self.config.feature.adj

        # 按股票分组, 减少 DB 查询次数
        missing_df = signals[
            signals.apply(
                lambda r: (r["ts_code"], r["signal_date"]) in missing_keys, axis=1
            )
        ]

        stock_groups = list(missing_df.groupby("ts_code"))
        total_stocks = len(stock_groups)

        for i, (ts_code, group) in enumerate(stock_groups, 1):
            if i % 200 == 0 or i == total_stocks:
                logger.info(f"  Feature extraction: {i}/{total_stocks} stocks, {len(rows)} rows")
            min_date = group["signal_date"].min()
            # 多取一些数据确保 lookback 够用
            daily = self.reader.get_stock_daily(
                ts_code, start_date="19900101", end_date=min_date, adj=adj,
            )
            if daily.empty:
                continue

            daily = daily.sort_values("trade_date").reset_index(drop=True)

            # 如果信号日期范围较大, 需要扩展查询
            max_date = group["signal_date"].max()
            if max_date > min_date:
                daily_ext = self.reader.get_stock_daily(
                    ts_code, start_date=min_date, end_date=max_date, adj=adj,
                )
                if not daily_ext.empty:
                    daily = pd.concat([daily, daily_ext]).drop_duplicates(
                        subset=["trade_date"]
                    ).sort_values("trade_date").reset_index(drop=True)

            date_list = daily["trade_date"].tolist()

            for _, row in group.iterrows():
                signal_date = row["signal_date"]
                # 找到 signal_date 或之前最近的交易日
                valid_dates = [d for d in date_list if d <= signal_date]
                if not valid_dates:
                    continue
                end_idx = date_list.index(valid_dates[-1])
                start_idx = max(0, end_idx - lookback + 1)
                window = daily.iloc[start_idx:end_idx + 1]

                feat = {"ts_code": ts_code, "signal_date": signal_date}
                for extractor in self.extractors:
                    feat.update(extractor.extract(ts_code, signal_date, window))
                rows.append(feat)

        return rows

    def _cache_path(self) -> Path:
        return self.cache_dir / f"{self.config.cache_key}.parquet"
