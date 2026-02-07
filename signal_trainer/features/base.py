from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseExtractor(ABC):
    """特征提取器基类"""

    @abstractmethod
    def extract(self, ts_code: str, signal_date: str, daily_df: pd.DataFrame) -> dict:
        """
        从 daily_df (已按 lookback 截取, 前复权) 中提取特征.
        返回 {特征名: 值} 字典.
        daily_df 按 trade_date 升序排列, 最后一行是 signal_date 当日或之前最近交易日.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """提取器名称, 用于配置和注册"""
