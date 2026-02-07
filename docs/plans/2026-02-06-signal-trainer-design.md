# Signal Trainer 架构设计

## 概述

消费 vector_tushare 回测框架导出的信号 CSV，训练二次分类模型预测信号后续涨跌。

- **数据源**: tushare-duckdb (环境变量 `DB_PATH`)
- **输入**: 信号导出协议 v1 CSV (`ts_code, signal_date`)
- **首个目标**: T+1 涨跌二分类 (LightGBM + 表格特征)
- **扩展方向**: 图像特征 + CNN, 时序模型, 多种标签定义

## 项目结构

```
signal_trainer/
├── configs/
│   └── default.yaml
├── signal_trainer/
│   ├── __init__.py
│   ├── cli.py                  # 命令行入口
│   ├── config.py               # 配置加载 (pydantic)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── signal_loader.py    # 读取信号 CSV
│   │   ├── feature_store.py    # 从 DuckDB 提取特征 + 缓存
│   │   └── label.py            # 标签计算（可插拔）
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py             # 特征提取基类
│   │   ├── technical.py        # 技术指标特征
│   │   └── money_flow.py       # 资金流特征
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # 模型基类
│   │   └── lgbm.py             # LightGBM 实现
│   ├── split/
│   │   ├── __init__.py
│   │   └── splitters.py        # 数据划分策略
│   └── pipeline.py             # 串联整个流程
├── pyproject.toml
└── README.md
```

## 数据流

```
信号CSV → 特征提取 → 标签计算 → 划分 → 训练/评估
```

1. **SignalLoader**: 读取 CSV, 得到 `[(ts_code, signal_date), ...]`
2. **FeatureStore**: 查询 signal_date 前 N 个交易日数据, 调用 Extractor 提取特征
3. **LabelCalculator**: 查询 signal_date 后价格, 计算标签
4. **Splitter**: 按时间或分层策略划分
5. **Model**: 训练 + 评估

### 图像扩展

表格和图像两条路径共享 SignalLoader 和 LabelCalculator, 分叉点在特征提取:

- 表格: FeatureStore → DataFrame → LightGBM
- 图像: ImageRenderer → PNG/Tensor → CNN (PyTorch Dataset)

## 核心接口

### BaseLabeler

```python
class BaseLabeler(ABC):
    @abstractmethod
    def compute(self, signals: pd.DataFrame, reader: DataReader) -> pd.Series:
        """输入信号列表, 返回每个信号的标签值"""
```

首个实现: `NextDayReturnLabeler` — T+1 收益率 > threshold → 1, 否则 → 0

### BaseExtractor

```python
class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, ts_code: str, end_date: str, lookback: int,
                daily_df: pd.DataFrame) -> dict:
        """返回 {特征名: 值} 字典"""
```

实现: `TechnicalExtractor`, `MoneyFlowExtractor`

### BaseModel

```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val): ...
    def predict(self, X) -> np.ndarray: ...
    def evaluate(self, X, y) -> dict: ...
```

### BaseSplitter

```python
class BaseSplitter(ABC):
    @abstractmethod
    def split(self, data, labels) -> tuple: ...
```

实现: `TimeBasedSplitter` (表格), `StratifiedSampler` (图像)

## 缓存设计

```
cache/
├── features/
│   └── {signal_file_stem}_{config_hash}.parquet
└── images/
    └── {signal_file_stem}_{config_hash}/
        ├── 000001.SZ_20240115.png
        └── ...
```

- **cache key**: 信号文件名 + 特征配置 hash
- **config_hash**: 由 lookback, extractors, adj 等参数决定
- **增量更新**: 只计算缺失信号的特征, 合并到已有缓存

## 配置文件

```yaml
signal_file: /path/to/exports/shaofu_20240101_20260205.csv
db_path: null  # null 读 DB_PATH 环境变量

label:
  type: next_day_return
  threshold: 0.0

feature:
  mode: tabular
  lookback: 30
  adj: qfq
  extractors:
    - technical
    - money_flow

split:
  method: time_based
  train_end: "20250630"

model:
  type: lgbm
  params:
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500

cache:
  enabled: true
  dir: cache/
```

CLI: `python -m signal_trainer --config configs/default.yaml`

## 训练/测试划分

- **表格特征**: 按时间切分 (train_end 之前训练, 之后测试)
- **图像特征**: 随机划分或按板块/指数/市值分层采样

## 关键约定

- 信号是盘后产生, signal_date 当日数据可作为特征
- 使用前复权 (qfq) 价格
- 标签计算需 signal_date 后至少 1 个交易日, 缺失则跳过
- 特征窗口默认 30 个交易日
