from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseExtractor


class TechnicalExtractor(BaseExtractor):
    """技术指标特征: MA, MACD, RSI, KDJ, 布林带, 量价"""

    @property
    def name(self) -> str:
        return "technical"

    def extract(self, ts_code: str, signal_date: str, daily_df: pd.DataFrame) -> dict:
        if daily_df.empty or len(daily_df) < 5:
            return {}

        close = daily_df["close"].values
        high = daily_df["high"].values
        low = daily_df["low"].values
        volume = daily_df["vol"].values
        last = close[-1]

        features = {}

        # --- 均线偏离 ---
        for n in (5, 10, 20):
            if len(close) >= n:
                ma = np.mean(close[-n:])
                features[f"ma{n}_bias"] = (last - ma) / ma

        # --- MACD ---
        if len(close) >= 26:
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            dif = ema12 - ema26
            dea = self._ema_from_series(dif, 9)
            features["macd_dif"] = dif[-1]
            features["macd_dea"] = dea[-1]
            features["macd_hist"] = 2 * (dif[-1] - dea[-1])

        # --- RSI ---
        for n in (6, 12):
            if len(close) > n:
                features[f"rsi_{n}"] = self._rsi(close, n)

        # --- KDJ ---
        if len(close) >= 9:
            k, d, j = self._kdj(high, low, close)
            features["kdj_k"] = k
            features["kdj_d"] = d
            features["kdj_j"] = j

        # --- 布林带 ---
        if len(close) >= 20:
            ma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            if std20 > 0:
                features["boll_pos"] = (last - ma20) / (2 * std20)

        # --- 量比 ---
        if len(volume) >= 6:
            vol_ma5 = np.mean(volume[-6:-1])
            if vol_ma5 > 0:
                features["vol_ratio"] = volume[-1] / vol_ma5

        # --- 涨跌幅 ---
        for n in (1, 3, 5, 10):
            if len(close) > n:
                features[f"ret_{n}d"] = (close[-1] - close[-1 - n]) / close[-1 - n]

        # --- 振幅 ---
        if len(high) >= 5:
            features["amplitude_5d"] = (np.max(high[-5:]) - np.min(low[-5:])) / close[-1]

        # --- 换手率趋势 (如果有 turnover_rate) ---
        if "turnover_rate" in daily_df.columns:
            tr = daily_df["turnover_rate"].values
            if len(tr) >= 5:
                features["turnover_5d_mean"] = np.mean(tr[-5:])
                features["turnover_trend"] = np.mean(tr[-3:]) - np.mean(tr[-5:])

        return features

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _ema_from_series(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> float:
        deltas = np.diff(close[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             n: int = 9) -> tuple[float, float, float]:
        ln = high[-n:]
        ll = low[-n:]
        highest = np.max(ln)
        lowest = np.min(ll)
        if highest == lowest:
            rsv = 50.0
        else:
            rsv = (close[-1] - lowest) / (highest - lowest) * 100
        # 简化: 取最近值作为 K, D
        k = rsv
        d = rsv
        j = 3 * k - 2 * d
        return k, d, j
