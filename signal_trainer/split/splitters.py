from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class BaseSplitter(ABC):
    @abstractmethod
    def split(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        signals: pd.DataFrame,
    ) -> SplitResult:
        """
        划分训练集和测试集.
        signals 用于获取 signal_date 等元数据辅助划分.
        """


class TimeBasedSplitter(BaseSplitter):
    """按 signal_date 时间切分"""

    def __init__(self, train_end: str):
        self.train_end = train_end

    def split(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        signals: pd.DataFrame,
    ) -> SplitResult:
        train_mask = signals["signal_date"] <= self.train_end
        test_mask = ~train_mask

        return SplitResult(
            X_train=features.loc[train_mask].reset_index(drop=True),
            y_train=labels.loc[train_mask].reset_index(drop=True),
            X_test=features.loc[test_mask].reset_index(drop=True),
            y_test=labels.loc[test_mask].reset_index(drop=True),
        )


class RandomSplitter(BaseSplitter):
    """随机划分 (图像方法用)"""

    def __init__(self, test_ratio: float = 0.2, seed: int = 42):
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        signals: pd.DataFrame,
    ) -> SplitResult:
        rng = np.random.RandomState(self.seed)
        n = len(features)
        indices = rng.permutation(n)
        split_idx = int(n * (1 - self.test_ratio))

        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        return SplitResult(
            X_train=features.iloc[train_idx].reset_index(drop=True),
            y_train=labels.iloc[train_idx].reset_index(drop=True),
            X_test=features.iloc[test_idx].reset_index(drop=True),
            y_test=labels.iloc[test_idx].reset_index(drop=True),
        )


SPLITTERS = {
    "time_based": TimeBasedSplitter,
    "random": RandomSplitter,
}


def create_splitter(split_config) -> BaseSplitter:
    if split_config.method == "time_based":
        return TimeBasedSplitter(train_end=split_config.train_end)
    elif split_config.method == "random":
        return RandomSplitter(test_ratio=split_config.test_ratio)
    else:
        raise ValueError(f"Unknown split method: {split_config.method}")
