# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run pipeline (DB_PATH env var required)
DB_PATH=/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db \
  python -m signal_trainer -c configs/default.yaml

# Verbose logging
python -m signal_trainer -c configs/gaf.yaml --verbose

# Run tests (need DB_PATH)
DB_PATH=/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db pytest tests/

# macOS OpenMP workaround (required for torch imports)
KMP_DUPLICATE_LIB_OK=TRUE python -m signal_trainer -c configs/multi_scale.yaml
```

## Architecture

The project is a configurable ML pipeline for stock signal prediction. A YAML config selects between two parallel paths through a 6-step pipeline:

**Tabular path** (`mode: tabular`): Signal CSV → FeatureStore (technical/money_flow extractors → parquet cache) → LightGBM classification

**Image path** (`mode: image`): Signal CSV → GAFStore (OHLCV → 10-channel Gramian Angular Field tensors → .pt cache) → ResNet-18 regression

### Config → Pipeline flow

`Config.from_yaml()` produces a Pydantic `Config` with nested models: `FeatureConfig`, `LabelConfig`, `SplitConfig`, `ModelConfig`, `CacheConfig`. The `Pipeline` class reads `feature.mode` to dispatch to `_run_tabular()` or `_run_image()`.

### Registry patterns

Three factory registries drive extensibility:
- **Models** (`models/__init__.py`): `{"lgbm": LGBMModel, "cnn": CNNModel}` — `create_model(model_config)`
- **Labelers** (`data/label.py`): `{"next_day_return", "max_return_n_days", "risk_return"}` — `create_labeler(label_config)`
- **Splitters** (`split/splitters.py`): `{"time_based", "random"}` — `create_splitter(split_config)`

Feature extractors use a separate registry in `data/feature_store.py`: `{"technical": TechnicalExtractor, "money_flow": MoneyFlowExtractor}`.

### GAF (image mode) specifics

5 OHLCV series × (GASF + GADF) = 10 channels per image. Spatial size equals lookback window (e.g., 120×120). Multi-scale mode (`lookback: [30, 60, 120]`) runs a shared ResNet encoder on each scale independently, concatenates 512-dim features per scale, then feeds through a regressor. `AdaptiveAvgPool2d(1)` handles varying spatial sizes. Each `.pt` tensor is cached per (ts_code, signal_date, lookback) with hash-based directory naming.

### Caching

- Features: `cache/features/{signal_stem}_{hash}.parquet` — incremental append
- GAF tensors: `cache/gaf/{signal_stem}_{hash}/{code}_{date}.pt` — per-sample files, separate dirs per lookback scale

### Data dependency

All market data comes from `tushare-db` (`DataReader`) backed by a local DuckDB file. The `DB_PATH` environment variable is required unless `db_path` is set in the YAML config.

## Configs

| Config | Mode | Model | Targets |
|--------|------|-------|---------|
| `default.yaml` | tabular | LightGBM | binary (next-day return > threshold) |
| `gaf.yaml` | image | GAFResNet | single regression |
| `risk_return.yaml` | image | GAFResNet | dual (max_return, max_drawdown) |
| `multi_scale.yaml` | image | MultiScaleGAFResNet | dual, 3 scales [30,60,120] |
