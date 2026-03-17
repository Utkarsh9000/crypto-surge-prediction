"""
Walk-forward backtesting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

import pandas as pd

from .evaluate import evaluate as evaluate_model
from .features import (
    FEATURE_COLS,
    LABEL_COL,
    NEXT_RETURN_COL,
    SURGE_HORIZON_DAYS,
    SURGE_THRESHOLD,
    build_features,
)
from .train import train_models


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--top-k", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=SURGE_HORIZON_DAYS)
    parser.add_argument("--surge-threshold", type=float, default=SURGE_THRESHOLD)
    parser.add_argument("--calib-ratio", type=float, default=0.2)
    return parser.parse_args()


def _prepare_features(
    df: pd.DataFrame, horizon: int, surge_threshold: float
) -> pd.DataFrame:
    if set(FEATURE_COLS).issubset(df.columns) and LABEL_COL in df.columns:
        out = df.copy()
    else:
        out = build_features(
            df,
            horizon=horizon,
            surge_threshold=surge_threshold,
            include_labels=True,
        )
    out["date"] = pd.to_datetime(out["date"])
    return out


def walk_forward(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    horizon: int,
    surge_threshold: float,
    calib_ratio: float,
    top_k: float,
) -> pd.DataFrame:
    df = df.sort_values("date")
    start = df["date"].min()
    end = df["date"].max()

    windows: List[dict[str, Any]] = []
    cur = start

    while True:
        train_start = cur
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        if test_end > end:
            break

        train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
        if train_df.empty or test_df.empty:
            cur = cur + pd.Timedelta(days=step_days)
            continue

        bundle = train_models(
            train_df,
            test_ratio=0.0,
            calib_ratio=calib_ratio,
            horizon=horizon,
            surge_threshold=surge_threshold,
        )

        metrics, _ = evaluate_model(
            df=test_df,
            classifier=bundle["classifier"],
            regressor=bundle["regressor"],
            feature_cols=bundle["metadata"]["feature_cols"],
            horizon=horizon,
            threshold=bundle["metrics"].get("threshold", 0.5),
            top_k=top_k,
        )

        windows.append(
            {
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                **metrics,
            }
        )

        cur = cur + pd.Timedelta(days=step_days)

    return pd.DataFrame(windows)


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    df = _prepare_features(
        df, horizon=args.horizon, surge_threshold=args.surge_threshold
    )

    report = walk_forward(
        df=df,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        horizon=args.horizon,
        surge_threshold=args.surge_threshold,
        calib_ratio=args.calib_ratio,
        top_k=args.top_k,
    )

    if report.empty:
        raise ValueError("No windows produced. Try smaller train/test windows.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"Wrote walk-forward report to: {args.out}")


if __name__ == "__main__":
    main()
