from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import torch

from comb2 import ComboBase, LoaderConfig
from comb2_pcmaster import BacktestNode, DailyBacktest
from factorsim import IndexMask

ORGANIZE_ROOT = Path(__file__).resolve().parent
organize_config_spec = importlib.util.spec_from_file_location("comb2_organize_config", ORGANIZE_ROOT / "config.py")
organize_config_module = importlib.util.module_from_spec(organize_config_spec)
assert organize_config_spec.loader is not None
organize_config_spec.loader.exec_module(organize_config_module)


class Node:
    def __init__(self, config: dict):
        instsz = len(IndexMask().code)
        self.alpha = torch.zeros(instsz, dtype=config["loader"]["dtype"])
        self.alpha_history: dict[int, torch.Tensor] = {}

        for section in ("paths", "runtime", "model", "output", "defaults"):
            for key, value in config[section].items():
                setattr(self, key, value)

        self.loader_config = LoaderConfig(**config["loader"])


def print_daily_metrics(metrics: dict):
    
    columns = [
        ("date", str(metrics["date"]), 10),
        ("pnl", f"{metrics['pnl']:.2f}", 14),
        ("total_asset", f"{metrics['total_asset']:.2f}", 16),
        ("trade_cost", f"{metrics['trade_cost']:.2f}", 14),
        ("reserve_cash", f"{metrics['reserve_cash']:.2f}", 16),
        ("tvr", f"{metrics['tvr']:.4f}", 10),
        ("long_num", str(metrics["long_num"]), 10),
    ]
    header = " ".join(f"{name:<{width}}" for name, _, width in columns)
    row = " ".join(f"{value:<{width}}" for _, value, width in columns)
    print("[BACKTEST]")
    print(header)
    print(row)


def build_strategy_file() -> Path:
    return Path(__file__).resolve().parent / "alpha_strategy.py"


def build_backtest_node(strategy_path: Path, organize_config: dict) -> BacktestNode:
    strategy_config = organize_config["strategy"]
    backtest_config = organize_config["backtest"]
    output_path = Path(backtest_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    return BacktestNode(
        start_ds=int(strategy_config["start_ds"]),
        end_ds=int(strategy_config["end_ds"]),
        output_path=str(output_path),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        cash=float(backtest_config["cash"]),
        fee_rate=float(backtest_config["fee_rate"]),
        reserve_cash=float(backtest_config["reserve_cash"]),
        daily_metrics_file=backtest_config.get("daily_metrics_file", "daily_metrics.csv"),
        verbose=bool(backtest_config["verbose"]),
        universe=backtest_config.get("universe", "base"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config_flag or args.config
    organize_config = organize_config_module.load_config(config_path)
    combo_config = organize_config["combo"]

    node = Node(combo_config)
    combo = ComboBase(node)
    codes = pd.Index([str(code).zfill(6) for code in IndexMask().code])

    strategy_path = build_strategy_file()
    backtest_node = build_backtest_node(strategy_path, organize_config)
    backtest = DailyBacktest(backtest_node)

    for date in sorted(backtest.vwap_data.index):
        combo.Combine(int(date))
        alpha = node.alpha.detach().cpu().to(dtype=node.alpha.dtype).numpy()
        metrics = backtest.step(int(date), pd.Series(alpha, index=codes))
        print_daily_metrics(metrics)

    backtest.finalize()


if __name__ == "__main__":
    main()
