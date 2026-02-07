from __future__ import annotations

import numpy as np
import pandas as pd
from tushare_db import DataReader

from .base import BaseExtractor


class MoneyFlowExtractor(BaseExtractor):
    """资金流向特征: 基于 moneyflow 和 moneyflow_dc 表"""

    def __init__(self, reader: DataReader):
        self._reader = reader

    @property
    def name(self) -> str:
        return "money_flow"

    def extract(self, ts_code: str, signal_date: str, daily_df: pd.DataFrame) -> dict:
        features = {}

        if daily_df.empty:
            return features

        start_date = daily_df["trade_date"].iloc[0]
        end_date = daily_df["trade_date"].iloc[-1]

        # --- moneyflow (标准接口, 2010 后有数据) ---
        mf = self._reader.get_moneyflow(
            ts_code=ts_code, start_date=start_date, end_date=end_date,
        )
        if not mf.empty:
            mf = mf.sort_values("trade_date")
            last = mf.iloc[-1]
            # 主力净流入 (大单+超大单)
            buy_lg = float(last.get("buy_lg_amount", 0) or 0)
            sell_lg = float(last.get("sell_lg_amount", 0) or 0)
            buy_elg = float(last.get("buy_elg_amount", 0) or 0)
            sell_elg = float(last.get("sell_elg_amount", 0) or 0)
            features["main_net_flow"] = (buy_lg + buy_elg) - (sell_lg + sell_elg)

            # 散户净流入 (小单)
            buy_sm = float(last.get("buy_sm_amount", 0) or 0)
            sell_sm = float(last.get("sell_sm_amount", 0) or 0)
            features["retail_net_flow"] = buy_sm - sell_sm

        return features
