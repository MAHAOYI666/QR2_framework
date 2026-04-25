import argparse
import os

import pandas as pd

from factorsim.config import Cache_Path
from run_dailypnl import loader, pnl


def resolve_ashare_data_path(produce_mode: bool) -> str:
    if produce_mode:
        return os.path.join(Cache_Path, "AshareCache")
    return "/root/CacheData/AshareCache"


def validate_parquet(path: str, arg_name: str) -> None:
    if not path.endswith(".parquet"):
        raise ValueError(f"{arg_name} must be a parquet file, got: {path}")


def calc_pnl_corr(
    signal_path_a: str,
    signal_path_b: str,
    produce_mode: bool = False,
    tradecost_ratio: float = 0.0,
    booksize: float = 1e7,
) -> dict:
    validate_parquet(signal_path_a, "signal_path_a")
    validate_parquet(signal_path_b, "signal_path_b")

    ashare_data_path = resolve_ashare_data_path(produce_mode)

    signal_a = loader(signal_path_a)
    signal_b = loader(signal_path_b)

    pnl_a = pnl(
        signal_a,
        ashare_data_path=ashare_data_path,
        booksize=booksize,
        tradecost_ratio=tradecost_ratio,
    )
    pnl_b = pnl(
        signal_b,
        ashare_data_path=ashare_data_path,
        booksize=booksize,
        tradecost_ratio=tradecost_ratio,
    )

    aligned = pd.concat(
        [
            pnl_a["pnl"].rename("pnl_a"),
            pnl_b["pnl"].rename("pnl_b"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError("No overlapping non-NaN pnl observations for correlation.")

    corr_value = aligned["pnl_a"].corr(aligned["pnl_b"])

    return {
        "corr": float(corr_value),
        "n_obs": int(len(aligned)),
        "start_ds": int(aligned.index.min()),
        "end_ds": int(aligned.index.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute daily pnl correlation between two parquet signals."
    )
    parser.add_argument("signal_path_a", type=str, help="First parquet signal path")
    parser.add_argument("signal_path_b", type=str, help="Second parquet signal path")
    parser.add_argument(
        "--produce_mode",
        action="store_true",
        default=False,
        help="Use production cache path",
    )
    parser.add_argument(
        "--tradecost_ratio",
        type=float,
        default=0.0,
        help="Transaction cost ratio (actual cost = ratio × 0.003)",
    )
    parser.add_argument(
        "--booksize",
        type=float,
        default=1e7,
        help="Book size for pnl calculation",
    )
    args = parser.parse_args()

    result = calc_pnl_corr(
        signal_path_a=args.signal_path_a,
        signal_path_b=args.signal_path_b,
        produce_mode=args.produce_mode,
        tradecost_ratio=args.tradecost_ratio,
        booksize=args.booksize,
    )

    print(f"pnl_corr: {result['corr']:.8f}")
    print(f"n_obs: {result['n_obs']}")
    print(f"date_range: {result['start_ds']} - {result['end_ds']}")


if __name__ == "__main__":
    main()
