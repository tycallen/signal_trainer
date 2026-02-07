from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .config import Config
from .pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Signal Trainer - 信号二次分类模型")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="实验配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    args = parser.parse_args()

    # 显式配置 signal_trainer logger (basicConfig 可能被 tushare_db 抢先)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    st_logger = logging.getLogger("signal_trainer")
    st_logger.setLevel(log_level)
    st_logger.propagate = False  # 不向 root logger 传播, 避免重复输出
    if not st_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
        ))
        st_logger.addHandler(handler)

    config = Config.from_yaml(args.config)
    pipeline = Pipeline(config)
    metrics = pipeline.run()

    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
