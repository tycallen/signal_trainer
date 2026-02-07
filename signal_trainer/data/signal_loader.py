from __future__ import annotations

from pathlib import Path

import pandas as pd


class SignalLoader:
    """读取信号导出协议 v1 格式的 CSV 文件"""

    def __init__(self, signal_file: str | Path):
        self.signal_file = Path(signal_file)

    def load(self) -> pd.DataFrame:
        """返回 DataFrame, 列: ts_code, signal_date"""
        df = pd.read_csv(
            self.signal_file,
            dtype={"ts_code": str, "signal_date": str},
        )
        assert set(df.columns) >= {"ts_code", "signal_date"}, (
            f"信号文件缺少必要字段, 实际列: {list(df.columns)}"
        )
        df = df[["ts_code", "signal_date"]].drop_duplicates().reset_index(drop=True)
        return df
