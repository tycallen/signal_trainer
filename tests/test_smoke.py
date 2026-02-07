"""端到端冒烟测试: 用少量真实信号跑通完整 pipeline"""
import os
import tempfile
from pathlib import Path

DB_PATH = "/Users/allen/workspace/python/stock/Tushare-DuckDB/tushare.db"


def test_pipeline_smoke():
    os.environ["DB_PATH"] = DB_PATH

    # 创建临时信号文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("ts_code,signal_date\n")
        f.write("000001.SZ,20241015\n")
        f.write("600519.SH,20241015\n")
        f.write("000002.SZ,20241016\n")
        signal_file = f.name

    # 创建配置
    from signal_trainer.config import Config

    config = Config(
        signal_file=signal_file,
        label={"type": "next_day_return", "threshold": 0.0},
        feature={"mode": "tabular", "lookback": 30, "adj": "qfq", "extractors": ["technical"]},
        split={"method": "time_based", "train_end": "20241015"},
        model={"type": "lgbm", "params": {"n_estimators": 10, "num_leaves": 8, "verbosity": -1}},
        cache={"enabled": False, "dir": tempfile.mkdtemp()},
    )

    from signal_trainer.pipeline import Pipeline

    pipeline = Pipeline(config)
    metrics = pipeline.run()

    print(f"Metrics: {metrics}")
    assert "accuracy" in metrics
    assert "auc" in metrics

    Path(signal_file).unlink()


if __name__ == "__main__":
    test_pipeline_smoke()
