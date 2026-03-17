"""
Feature engineering and surge labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import bollinger, ema, macd, rsi


SURGE_THRESHOLD = 0.15  # 15% next-horizon return
SURGE_HORIZON_DAYS = 7
LABEL_COL = "surge_label"
NEXT_RETURN_COL = "next_return"

FEATURE_COLS = [
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "ret_14d",
    "ret_30d",
    "log_ret_1d",
    "vol_7d",
    "vol_14d",
    "vol_30d",
    "price_sma_7",
    "price_sma_14",
    "price_sma_30",
    "price_sma_90",
    "price_z_30",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width_20",
    "bb_z_20",
    "volume_ret_1d",
    "volume_z_30",
    "mcap_log",
    "mcap_ret_7d",
    "market_ret_1d",
    "market_ret_7d",
    "market_vol_7d",
    "btc_ret_1d",
    "btc_ret_7d",
    "btc_vol_7d",
    "btc_dominance",
    "mcap_rank_pct",
    "rel_ret_1d",
    "rel_ret_7d",
]


def _add_coin_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("date").copy()

    price = grp["price"].astype(float)
    volume = grp["volume"].astype(float)
    mcap = grp["market_cap"].astype(float)

    log_price = np.log(price.replace(0, np.nan))
    log_ret_1d = log_price.diff()

    grp["ret_1d"] = price.pct_change()
    grp["ret_3d"] = price.pct_change(3)
    grp["ret_7d"] = price.pct_change(7)
    grp["ret_14d"] = price.pct_change(14)
    grp["ret_30d"] = price.pct_change(30)
    grp["log_ret_1d"] = log_ret_1d

    grp["vol_7d"] = log_ret_1d.rolling(7).std()
    grp["vol_14d"] = log_ret_1d.rolling(14).std()
    grp["vol_30d"] = log_ret_1d.rolling(30).std()

    sma_7 = price.rolling(7).mean()
    sma_14 = price.rolling(14).mean()
    sma_30 = price.rolling(30).mean()
    sma_90 = price.rolling(90).mean()
    grp["price_sma_7"] = price / sma_7 - 1
    grp["price_sma_14"] = price / sma_14 - 1
    grp["price_sma_30"] = price / sma_30 - 1
    grp["price_sma_90"] = price / sma_90 - 1

    price_std_30 = price.rolling(30).std()
    grp["price_z_30"] = (price - sma_30) / price_std_30

    grp["ema_12"] = ema(price, 12)
    grp["ema_26"] = ema(price, 26)
    grp["rsi_14"] = rsi(price, 14)
    macd_line, macd_signal, macd_hist = macd(price, 12, 26, 9)
    grp["macd"] = macd_line
    grp["macd_signal"] = macd_signal
    grp["macd_hist"] = macd_hist

    bb_mid, bb_upper, bb_lower, bb_std = bollinger(price, 20, 2.0)
    grp["bb_width_20"] = (bb_upper - bb_lower) / bb_mid
    grp["bb_z_20"] = (price - bb_mid) / bb_std

    grp["volume_ret_1d"] = volume.pct_change()
    vol_mean_30 = volume.rolling(30).mean()
    vol_std_30 = volume.rolling(30).std()
    grp["volume_z_30"] = (volume - vol_mean_30) / vol_std_30

    grp["mcap_log"] = np.log(mcap.replace(0, np.nan))
    grp["mcap_ret_7d"] = mcap.pct_change(7)

    return grp


def build_features(
    df: pd.DataFrame,
    horizon: int = SURGE_HORIZON_DAYS,
    surge_threshold: float = SURGE_THRESHOLD,
    include_labels: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["coin_id", "date"])

    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["market_cap"] = df["market_cap"].astype(float)

    df = df.groupby("coin_id", group_keys=False).apply(_add_coin_features)

    df["market_ret_1d"] = df.groupby("date")["ret_1d"].transform("mean")
    df["market_ret_7d"] = df.groupby("date")["ret_7d"].transform("mean")
    df["market_vol_7d"] = df.groupby("date")["vol_7d"].transform("mean")
    df["market_cap_total"] = df.groupby("date")["market_cap"].transform("sum")

    btc = df[df["coin_id"] == "bitcoin"][
        ["date", "ret_1d", "ret_7d", "vol_7d", "market_cap"]
    ].rename(
        columns={
            "ret_1d": "btc_ret_1d",
            "ret_7d": "btc_ret_7d",
            "vol_7d": "btc_vol_7d",
            "market_cap": "btc_market_cap",
        }
    )
    if btc.empty:
        df["btc_ret_1d"] = df["market_ret_1d"]
        df["btc_ret_7d"] = df["market_ret_7d"]
        df["btc_vol_7d"] = df["market_vol_7d"]
        df["btc_market_cap"] = 0.0
    else:
        df = df.merge(btc, on="date", how="left")
    df["btc_dominance"] = df["btc_market_cap"] / df["market_cap_total"]
    df["btc_dominance"] = df["btc_dominance"].fillna(0.0)

    df["mcap_rank_pct"] = df.groupby("date")["market_cap"].rank(
        ascending=False, pct=True
    )
    df["rel_ret_1d"] = df["ret_1d"] - df["market_ret_1d"]
    df["rel_ret_7d"] = df["ret_7d"] - df["market_ret_7d"]

    df = df.replace([np.inf, -np.inf], np.nan)

    required = list(FEATURE_COLS)
    if include_labels:
        df[NEXT_RETURN_COL] = df.groupby("coin_id")["price"].pct_change(-horizon)
        df[LABEL_COL] = (df[NEXT_RETURN_COL] >= surge_threshold).astype(int)
        required.append(NEXT_RETURN_COL)

    df = df.dropna(subset=required)
    if include_labels:
        df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features and labels")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=SURGE_HORIZON_DAYS)
    parser.add_argument("--surge-threshold", type=float, default=SURGE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    out = build_features(
        df,
        horizon=args.horizon,
        surge_threshold=args.surge_threshold,
        include_labels=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote features to: {args.out}")


if __name__ == "__main__":
    main()
