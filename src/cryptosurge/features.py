"""
Feature engineering and surge labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SURGE_THRESHOLD = 0.15  # 15% next-7-day return


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["coin_id", "date"])

    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["market_cap"] = df["market_cap"].astype(float)

    df["ret_1d"] = df.groupby("coin_id")["price"].pct_change()
    df["ret_7d"] = df.groupby("coin_id")["price"].pct_change(7)
    df["ret_30d"] = df.groupby("coin_id")["price"].pct_change(30)

    df["vol_7d"] = df.groupby("coin_id")["ret_1d"].rolling(7).std().reset_index(level=0, drop=True)
    df["vol_30d"] = df.groupby("coin_id")["ret_1d"].rolling(30).std().reset_index(level=0, drop=True)

    df["vol_z"] = (
        (df["volume"] - df.groupby("coin_id")["volume"].transform("mean"))
        / df.groupby("coin_id")["volume"].transform("std")
    )

    # Label: next-7-day return
    df["next_7d_return"] = df.groupby("coin_id")["price"].pct_change(-7)
    df["surge_label"] = (df["next_7d_return"] >= SURGE_THRESHOLD).astype(int)

    feature_cols = [
        "ret_1d",
        "ret_7d",
        "ret_30d",
        "vol_7d",
        "vol_30d",
        "vol_z",
    ]
    df = df.dropna(subset=feature_cols + ["surge_label"])
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features and labels")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    out = build_features(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote features to: {args.out}")


if __name__ == "__main__":
    main()
