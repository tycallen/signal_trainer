from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class LabelConfig(BaseModel):
    type: str = "next_day_return"
    threshold: float = 0.0
    # 扩展字段
    n: int = 1  # 用于 T+N 类标签

    def config_hash(self) -> str:
        key = f"{self.type}_{self.threshold}_{self.n}"
        return hashlib.md5(key.encode()).hexdigest()[:8]


class FeatureConfig(BaseModel):
    mode: str = "tabular"  # tabular | image
    lookback: int | list[int] = 30
    adj: str = "qfq"
    extractors: list[str] = Field(default_factory=lambda: ["technical", "money_flow"])

    @property
    def lookback_list(self) -> list[int]:
        """始终返回 list 形式的 lookback"""
        if isinstance(self.lookback, list):
            return self.lookback
        return [self.lookback]

    @property
    def is_multi_scale(self) -> bool:
        return isinstance(self.lookback, list) and len(self.lookback) > 1

    def config_hash(self) -> str:
        """根据影响特征生成的参数计算 hash"""
        lb = sorted(self.lookback) if isinstance(self.lookback, list) else self.lookback
        key = f"{self.mode}_{lb}_{self.adj}_{'_'.join(sorted(self.extractors))}"
        return hashlib.md5(key.encode()).hexdigest()[:8]


class SplitConfig(BaseModel):
    method: str = "time_based"  # time_based | random | stratified
    train_end: str = "20250630"
    test_ratio: float = 0.2  # random/stratified 时使用


class ModelConfig(BaseModel):
    type: str = "lgbm"
    params: dict = Field(default_factory=lambda: {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
    })


class CacheConfig(BaseModel):
    enabled: bool = True
    dir: str = "cache/"


class Config(BaseModel):
    signal_file: str
    db_path: str | None = None

    label: LabelConfig = Field(default_factory=LabelConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="after")
    def resolve_db_path(self) -> Config:
        if self.db_path is None:
            self.db_path = os.environ.get("DB_PATH")
            if not self.db_path:
                raise ValueError("db_path not set and DB_PATH env var is missing")
        return self

    @property
    def signal_file_stem(self) -> str:
        return Path(self.signal_file).stem

    @property
    def cache_key(self) -> str:
        return f"{self.signal_file_stem}_{self.feature.config_hash()}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
