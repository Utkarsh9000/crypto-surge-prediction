"""
Evaluate a trained model on labeled data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import (
    FEATURE_COLS,
    LABEL_COL,
    NEXT_RETURN_COL,
    SURGE_HORIZON_DAYS,
    SURGE_THRESHOLD,
    build_features,
)


def _load_bundle(model_dir: Path) -> dict | None:
    bundle_path = model_dir / "model_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None


def _safe_metric(fn, y_true, y_score) -> float | None:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return None


def _precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: float) -> float | None:
    if len(y_true) == 0:
        return None
    n = max(1, int(len(y_true) * k))
    idx = np.argsort(proba)[::-1][:n]
    return float(np.mean(y_true[idx]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate surge model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pred-out", type=Path, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--top-k", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=SURGE_HORIZON_DAYS)
    parser.add_argument("--surge-threshold", type=float, default=SURGE_THRESHOLD)
    return parser.parse_args()


def _filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df


def evaluate(
    df: pd.DataFrame,
    classifier,
    regressor,
    feature_cols: list[str],
    horizon: int,
    threshold: float,
    top_k: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    x = df[feature_cols]
    y_true = df[LABEL_COL].to_numpy()
    proba = classifier.predict_proba(x)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "auc": _safe_metric(roc_auc_score, y_true, proba),
        "pr_auc": _safe_metric(average_precision_score, y_true, proba),
        "accuracy": float(accuracy_score(y_true, preds)) if len(y_true) else None,
        "precision": float(precision_score(y_true, preds, zero_division=0))
        if len(y_true)
        else None,
        "recall": float(recall_score(y_true, preds, zero_division=0)) if len(y_true) else None,
        "f1": float(f1_score(y_true, preds, zero_division=0)) if len(y_true) else None,
        "brier": _safe_metric(brier_score_loss, y_true, proba),
        "precision_at_k": _precision_at_k(y_true, proba, top_k),
        "base_rate": float(np.mean(y_true)) if len(y_true) else None,
        "threshold": threshold,
        "top_k": top_k,
    }

    pred_return = regressor.predict(x)
    df = df.copy()
    df["surge_probability"] = proba
    df["surge_signal"] = preds
    df[f"predicted_{horizon}d_return"] = pred_return

    if NEXT_RETURN_COL in df.columns:
        signal_mask = df["surge_signal"] == 1
        metrics["signal_count"] = int(signal_mask.sum())
        metrics["signal_hit_rate"] = (
            float(df.loc[signal_mask, LABEL_COL].mean()) if signal_mask.any() else None
        )
        metrics["signal_avg_return"] = (
            float(df.loc[signal_mask, NEXT_RETURN_COL].mean())
            if signal_mask.any()
            else None
        )
        metrics["signal_avg_pred_return"] = (
            float(df.loc[signal_mask, f"predicted_{horizon}d_return"].mean())
            if signal_mask.any()
            else None
        )

        n = max(1, int(len(df) * top_k))
        top_idx = np.argsort(proba)[::-1][:n]
        metrics["top_k_avg_return"] = float(df.iloc[top_idx][NEXT_RETURN_COL].mean())
        metrics["top_k_avg_pred_return"] = float(
            df.iloc[top_idx][f"predicted_{horizon}d_return"].mean()
        )

    return metrics, df


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)

    bundle = _load_bundle(args.model_dir)
    if bundle:
        classifier = bundle["classifier"]
        regressor = bundle["regressor"]
        feature_cols = bundle["metadata"]["feature_cols"]
        threshold = bundle["metrics"].get("threshold", 0.5)
        horizon = bundle["metadata"].get("horizon", args.horizon)
        surge_threshold = bundle["metadata"].get("surge_threshold", args.surge_threshold)
    else:
        classifier = joblib.load(args.model_dir / "surge_classifier.joblib")
        regressor = joblib.load(args.model_dir / "return_regressor.joblib")
        feature_cols = FEATURE_COLS
        threshold = 0.5
        horizon = args.horizon
        surge_threshold = args.surge_threshold

    if not set(feature_cols).issubset(df.columns) or LABEL_COL not in df.columns:
        required_raw = {"date", "coin_id", "price", "volume", "market_cap"}
        if not required_raw.issubset(df.columns):
            missing = sorted(required_raw - set(df.columns))
            raise ValueError(
                "Input data is missing required columns: " + ", ".join(missing)
            )
        df = build_features(
            df,
            horizon=horizon,
            surge_threshold=surge_threshold,
            include_labels=True,
        )

    df = _filter_dates(df, args.start, args.end)
    if df.empty:
        raise ValueError("No rows remain after filtering by date.")

    metrics, scored = evaluate(
        df=df,
        classifier=classifier,
        regressor=regressor,
        feature_cols=feature_cols,
        horizon=horizon,
        threshold=threshold,
        top_k=args.top_k,
    )

    out_path = args.out or (args.model_dir / "metrics_eval.json")
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.pred_out:
        args.pred_out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.pred_out, index=False)

    print("Wrote evaluation metrics to:", out_path)


if __name__ == "__main__":
    main()
