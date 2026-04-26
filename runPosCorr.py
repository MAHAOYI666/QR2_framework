#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "comb2-metrics"))

from comb2_metrics import pos_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate mean daily position correlation for two parquet position files.")
    parser.add_argument("pos1", help="First position parquet path")
    parser.add_argument("pos2", help="Second position parquet path")
    parser.add_argument("--min-valid", type=int, default=1000, help="Minimum valid cross-section count per day")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corr = pos_corr(args.pos1, args.pos2, min_valid=args.min_valid)
    print(corr.mean())


if __name__ == "__main__":
    main()
