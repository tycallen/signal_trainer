from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd
from tushare_db import DataReader

logger = logging.getLogger(__name__)


class BaseLabeler(ABC):
    @abstractmethod
    def compute(self, signals: pd.DataFrame, reader: DataReader) -> pd.Series:
        """输入信号 DataFrame (ts_code, signal_date), 返回标签 Series (index 对齐)"""


class NextDayReturnLabeler(BaseLabeler):
    """T+1 涨跌二分类: 次日收益率 > threshold → 1, 否则 → 0"""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def compute(self, signals: pd.DataFrame, reader: DataReader) -> pd.Series:
        labels = pd.Series(index=signals.index, dtype="float64", name="label")
        stock_groups = list(signals.groupby("ts_code"))
        total_stocks = len(stock_groups)
        logger.info(f"Computing labels for {len(signals)} signals across {total_stocks} stocks")

        for i, (ts_code, group) in enumerate(stock_groups, 1):
            if i % 200 == 0 or i == total_stocks:
                filled = labels.notna().sum()
                logger.info(f"  Label progress: {i}/{total_stocks} stocks, {filled} labels computed")
            # 取该股票所有信号日期范围, 多查几天确保覆盖 T+1
            min_date = group["signal_date"].min()
            max_date = group["signal_date"].max()
            daily = reader.get_stock_daily(ts_code, start_date=min_date, adj="qfq")
            if daily.empty:
                continue

            daily = daily.sort_values("trade_date").reset_index(drop=True)
            date_list = daily["trade_date"].tolist()
            close_map = dict(zip(daily["trade_date"], daily["close"]))

            for idx, row in group.iterrows():
                signal_date = row["signal_date"]
                # 找 signal_date 在交易日列表中的位置
                if signal_date not in date_list:
                    # signal_date 不是交易日, 找下一个交易日
                    later = [d for d in date_list if d > signal_date]
                    if not later:
                        continue
                    t0 = later[0]
                    t0_idx = date_list.index(t0)
                else:
                    t0_idx = date_list.index(signal_date)
                    t0 = signal_date

                # T+1 日
                t1_idx = t0_idx + 1
                if t1_idx >= len(date_list):
                    continue

                t1 = date_list[t1_idx]
                ret = (close_map[t1] - close_map[t0]) / close_map[t0]
                labels.at[idx] = 1.0 if ret > self.threshold else 0.0

        return labels


class MaxReturnNDaysLabeler(BaseLabeler):
    """T+1 开盘买入, N 日内最高价最大涨幅 (回归: 连续值)"""

    def __init__(self, threshold: float = 0.05, n: int = 5):
        self.threshold = threshold  # 仅用于统计日志, 不影响标签值
        self.n = n

    def compute(self, signals: pd.DataFrame, reader: DataReader) -> pd.Series:
        labels = pd.Series(index=signals.index, dtype="float64", name="label")
        stock_groups = list(signals.groupby("ts_code"))
        total_stocks = len(stock_groups)
        logger.info(
            f"Computing max_return_{self.n}d regression labels "
            f"for {len(signals)} signals across {total_stocks} stocks"
        )

        for i, (ts_code, group) in enumerate(stock_groups, 1):
            if i % 200 == 0 or i == total_stocks:
                filled = labels.notna().sum()
                logger.info(f"  Label progress: {i}/{total_stocks} stocks, {filled} labels computed")

            min_date = group["signal_date"].min()
            daily = reader.get_stock_daily(ts_code, start_date=min_date, adj="qfq")
            if daily.empty:
                continue

            daily = daily.sort_values("trade_date").reset_index(drop=True)
            date_list = daily["trade_date"].tolist()

            for idx, row in group.iterrows():
                signal_date = row["signal_date"]
                # 找 signal_date 在交易日列表中的位置
                if signal_date not in date_list:
                    later = [d for d in date_list if d > signal_date]
                    if not later:
                        continue
                    t0_idx = date_list.index(later[0])
                else:
                    t0_idx = date_list.index(signal_date)

                # T+1 日开盘价作为买入价
                t1_idx = t0_idx + 1
                if t1_idx >= len(daily):
                    continue

                buy_price = daily.iloc[t1_idx]["open"]
                if buy_price <= 0:
                    continue

                # T+1 到 T+N 的最高价
                end_idx = min(t1_idx + self.n, len(daily))
                if end_idx <= t1_idx:
                    continue

                max_high = daily.iloc[t1_idx:end_idx]["high"].max()
                max_ret = (max_high - buy_price) / buy_price
                labels.at[idx] = max_ret  # 连续值

        return labels


class RiskReturnLabeler(BaseLabeler):
    """T+1 开盘买入, N 日内同时计算 max_return 和 max_drawdown"""

    def __init__(self, threshold: float = 0.05, n: int = 5):
        self.threshold = threshold
        self.n = n

    def compute(self, signals: pd.DataFrame, reader: DataReader) -> pd.DataFrame:
        """返回 DataFrame, 列: max_return, max_drawdown"""
        labels = pd.DataFrame(
            index=signals.index,
            columns=["max_return", "max_drawdown"],
            dtype="float64",
        )
        stock_groups = list(signals.groupby("ts_code"))
        total_stocks = len(stock_groups)
        logger.info(
            f"Computing risk_return_{self.n}d labels "
            f"for {len(signals)} signals across {total_stocks} stocks"
        )

        for i, (ts_code, group) in enumerate(stock_groups, 1):
            if i % 200 == 0 or i == total_stocks:
                filled = labels["max_return"].notna().sum()
                logger.info(f"  Label progress: {i}/{total_stocks} stocks, {filled} labels computed")

            min_date = group["signal_date"].min()
            daily = reader.get_stock_daily(ts_code, start_date=min_date, adj="qfq")
            if daily.empty:
                continue

            daily = daily.sort_values("trade_date").reset_index(drop=True)
            date_list = daily["trade_date"].tolist()

            for idx, row in group.iterrows():
                signal_date = row["signal_date"]
                if signal_date not in date_list:
                    later = [d for d in date_list if d > signal_date]
                    if not later:
                        continue
                    t0_idx = date_list.index(later[0])
                else:
                    t0_idx = date_list.index(signal_date)

                t1_idx = t0_idx + 1
                if t1_idx >= len(daily):
                    continue

                buy_price = daily.iloc[t1_idx]["open"]
                if buy_price <= 0:
                    continue

                end_idx = min(t1_idx + self.n, len(daily))
                if end_idx <= t1_idx:
                    continue

                window = daily.iloc[t1_idx:end_idx]
                max_high = window["high"].max()
                min_low = window["low"].min()
                labels.at[idx, "max_return"] = (max_high - buy_price) / buy_price
                labels.at[idx, "max_drawdown"] = (min_low - buy_price) / buy_price

        return labels


LABELERS = {
    "next_day_return": NextDayReturnLabeler,
    "max_return_n_days": MaxReturnNDaysLabeler,
    "risk_return": RiskReturnLabeler,
}


def create_labeler(label_config) -> BaseLabeler:
    cls = LABELERS.get(label_config.type)
    if cls is None:
        raise ValueError(f"Unknown labeler type: {label_config.type}")
    if label_config.type in ("max_return_n_days", "risk_return"):
        return cls(threshold=label_config.threshold, n=label_config.n)
    return cls(threshold=label_config.threshold)
