"""
Run the default alpha-mining scaffold on processed parquet data.

Example:
    python run_alpha_mining.py --split train --output outputs/alpha_mining_train
"""

import argparse
from pathlib import Path

import pandas as pd

from factors.alpha.miner import AlphaMiningConfig, mine_default_daily_factors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="outputs/alpha_mining")
    parser.add_argument("--windows", default="5,20", help="Comma-separated rolling windows.")
    parser.add_argument("--universe", default="all", choices=["all", "hs300", "csi500"])
    parser.add_argument("--factor-set", default="all", choices=["all", "daily", "fundamental"])
    parser.add_argument(
        "--fundamental",
        action="store_true",
        help="Read <split>_fundamental.parquet from processed-dir when available.",
    )
    parser.add_argument(
        "--fundamental-suffix",
        default="",
        help="Optional suffix for files like <split>_fundamental_<suffix>.parquet.",
    )
    parser.add_argument("--min-abs-ic", type=float, default=0.005)
    parser.add_argument("--min-coverage", type=float, default=0.55)
    parser.add_argument("--max-pair-corr", type=float, default=0.85)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    if args.fundamental:
        suffix = args.fundamental_suffix
        if not suffix and args.universe == "csi500":
            suffix = "000905_SH"
        tag = f"_{suffix}" if suffix else ""
        input_name = f"{args.split}_fundamental{tag}.parquet"
    else:
        input_name = f"{args.split}.parquet"
    df = pd.read_parquet(processed_dir / input_name)
    if args.universe == "hs300":
        if "index_weight" not in df.columns:
            raise KeyError("--universe hs300 requires an index_weight column")
        df = df[df["index_weight"] > 0].copy()
    elif args.universe == "csi500":
        weight_col = "csi500_index_weight" if "csi500_index_weight" in df.columns else "index_weight_000905_SH"
        if weight_col not in df.columns:
            raise KeyError("--universe csi500 requires csi500_index_weight or index_weight_000905_SH")
        df = df[df[weight_col] > 0].copy()
    windows = tuple(int(x) for x in args.windows.split(",") if x.strip())
    config = AlphaMiningConfig(
        min_abs_ic=args.min_abs_ic,
        min_coverage=args.min_coverage,
        max_pair_corr=args.max_pair_corr,
    )

    result = mine_default_daily_factors(
        df,
        output_dir=args.output,
        windows=windows,
        factor_set=args.factor_set,
        config=config,
    )
    print(f"Input: {processed_dir / input_name}")
    print(f"Factor set: {args.factor_set}")
    print(f"Universe rows: {len(df)}, stocks: {df['ts_code'].nunique()}, dates: {df['trade_date'].nunique()}")
    print(f"Evaluated {len(result.summary)} candidates.")
    print(f"Selected {len(result.selected)} factors: {', '.join(result.selected)}")
    if not result.summary.empty:
        print(result.summary[["factor", "RankIC_mean", "RankIC_IR", "coverage", "score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
