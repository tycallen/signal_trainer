from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from .base import BaseModel

logger = logging.getLogger(__name__)


class LGBMModel(BaseModel):
    def __init__(self, params: dict | None = None):
        default_params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        self.model = lgb.LGBMClassifier(**self.params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_kwargs)
        logger.info(f"Model trained with {len(X_train)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, zero_division=0),
        }
        if len(set(y)) > 1:
            metrics["auc"] = roc_auc_score(y, y_prob)
        else:
            metrics["auc"] = float("nan")

        logger.info(f"Evaluation: {metrics}")
        logger.info(f"\n{classification_report(y, y_pred, zero_division=0)}")
        return metrics

    def feature_importance(self) -> pd.Series | None:
        if self.model is None:
            return None
        importance = self.model.feature_importances_
        names = self.model.feature_name_
        return pd.Series(importance, index=names).sort_values(ascending=False)


MODELS = {
    "lgbm": LGBMModel,
}


def create_model(model_config) -> BaseModel:
    cls = MODELS.get(model_config.type)
    if cls is None:
        raise ValueError(f"Unknown model type: {model_config.type}")
    return cls(params=model_config.params)
