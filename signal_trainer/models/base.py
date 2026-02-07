from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回正类概率"""

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]: ...

    @abstractmethod
    def feature_importance(self) -> pd.Series | None:
        """返回特征重要性, 如果模型支持的话"""
