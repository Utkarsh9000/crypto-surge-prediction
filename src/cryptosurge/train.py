"""
Train surge classification and growth regression models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
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


def _safe_metric(fn, y_true, y_score) -> float | None:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return None


def _time_split_by_date(
    df: pd.DataFrame, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    df = df.sort_values("date")
    dates = np.array(sorted(df["date"].unique()))
    if len(dates) < 3:
        return df, df.iloc[0:0], df["date"].max()
    cutoff_idx = max(1, int(len(dates) * (1 - test_ratio)))
    cutoff_date = dates[cutoff_idx - 1]
    train_df = df[df["date"] <= cutoff_date]
    test_df = df[df["date"] > cutoff_date]
    return train_df, test_df, cutoff_date


def _calibration_split(
    df: pd.DataFrame, calib_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    df = df.sort_values("date")
    dates = np.array(sorted(df["date"].unique()))
    if len(dates) < 3:
        return df, df.iloc[0:0], df["date"].max()
    cutoff_idx = max(1, int(len(dates) * (1 - calib_ratio)))
    cutoff_date = dates[cutoff_idx - 1]
    train_df = df[df["date"] <= cutoff_date]
    calib_df = df[df["date"] > cutoff_date]
    return train_df, calib_df, cutoff_date


def _make_sample_weights(y: pd.Series) -> np.ndarray:
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return np.ones(len(y), dtype=float)
    weight_pos = neg / pos
    return np.where(y == 1, weight_pos, 1.0)


def _tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    best_score = -1.0
    best_t = 0.5
    for t in np.linspace(0.1, 0.9, 17):
        preds = (proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def _precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: float = 0.05) -> float | None:
    if len(y_true) == 0:
        return None
    n = max(1, int(len(y_true) * k))
    idx = np.argsort(proba)[::-1][:n]
    return float(np.mean(y_true[idx]))


def _maybe_rename_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if NEXT_RETURN_COL not in df.columns and "next_7d_return" in df.columns:
        df = df.rename(columns={"next_7d_return": NEXT_RETURN_COL})
    return df


def train_models(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    calib_ratio: float = 0.2,
    horizon: int = SURGE_HORIZON_DAYS,
    surge_threshold: float = SURGE_THRESHOLD,
) -> dict[str, Any]:
    if not set(FEATURE_COLS).issubset(df.columns):
        df = build_features(
            df,
            horizon=horizon,
            surge_threshold=surge_threshold,
            include_labels=True,
        )
    df = _maybe_rename_legacy_columns(df)

    train_df, test_df, test_cutoff = _time_split_by_date(df, test_ratio=test_ratio)
    train_inner, calib_df, calib_cutoff = _calibration_split(train_df, calib_ratio=calib_ratio)

    x_train = train_inner[FEATURE_COLS]
    y_train = train_inner[LABEL_COL]

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.1,
        random_state=7,
    )
    clf.fit(x_train, y_train, sample_weight=_make_sample_weights(y_train))

    classifier = clf
    threshold = 0.5
    if not calib_df.empty and calib_df[LABEL_COL].nunique() > 1:
        calibrator = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        calibrator.fit(calib_df[FEATURE_COLS], calib_df[LABEL_COL])
        classifier = calibrator
        calib_proba = classifier.predict_proba(calib_df[FEATURE_COLS])[:, 1]
        threshold = _tune_threshold(calib_df[LABEL_COL].to_numpy(), calib_proba)

    x_test = test_df[FEATURE_COLS] if not test_df.empty else test_df
    y_test = test_df[LABEL_COL] if not test_df.empty else pd.Series(dtype=int)

    proba = (
        classifier.predict_proba(x_test)[:, 1]
        if not test_df.empty
        else np.array([], dtype=float)
    )
    preds = (proba >= threshold).astype(int) if len(proba) else np.array([], dtype=int)

    metrics = {
        "auc": _safe_metric(roc_auc_score, y_test, proba),
        "pr_auc": _safe_metric(average_precision_score, y_test, proba),
        "accuracy": float(accuracy_score(y_test, preds)) if len(preds) else None,
        "precision": float(precision_score(y_test, preds, zero_division=0))
        if len(preds)
        else None,
        "recall": float(recall_score(y_test, preds, zero_division=0)) if len(preds) else None,
        "f1": float(f1_score(y_test, preds, zero_division=0)) if len(preds) else None,
        "brier": _safe_metric(brier_score_loss, y_test, proba),
        "precision_at_5pct": _precision_at_k(
            y_test.to_numpy() if len(y_test) else np.array([]), proba, k=0.05
        ),
        "base_rate": float(np.mean(y_test)) if len(y_test) else None,
        "threshold": threshold,
    }

    reg = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.1,
        random_state=7,
    )
    reg.fit(train_df[FEATURE_COLS], train_df[NEXT_RETURN_COL])
    reg_preds = reg.predict(x_test) if not test_df.empty else np.array([], dtype=float)
    metrics["mae_return"] = (
        float(mean_absolute_error(test_df[NEXT_RETURN_COL], reg_preds))
        if len(reg_preds)
        else None
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": FEATURE_COLS,
        "horizon": horizon,
        "surge_threshold": surge_threshold,
        "train_end": train_df["date"].max().isoformat() if not train_df.empty else None,
        "test_start": test_df["date"].min().isoformat() if not test_df.empty else None,
        "test_end": test_df["date"].max().isoformat() if not test_df.empty else None,
        "test_cutoff": pd.Timestamp(test_cutoff).isoformat() if test_cutoff is not None else None,
        "calib_cutoff": pd.Timestamp(calib_cutoff).isoformat() if calib_cutoff is not None else None,
    }

    return {
        "classifier": classifier,
        "regressor": reg,
        "metrics": metrics,
        "metadata": metadata,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surge models")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--calib-ratio", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=SURGE_HORIZON_DAYS)
    parser.add_argument("--surge-threshold", type=float, default=SURGE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    out = train_models(
        df,
        test_ratio=args.test_ratio,
        calib_ratio=args.calib_ratio,
        horizon=args.horizon,
        surge_threshold=args.surge_threshold,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(out["classifier"], args.model_dir / "surge_classifier.joblib")
    joblib.dump(out["regressor"], args.model_dir / "return_regressor.joblib")
    joblib.dump(out["metrics"], args.model_dir / "metrics.joblib")
    joblib.dump(out, args.model_dir / "model_bundle.joblib")
    (args.model_dir / "metrics.json").write_text(
        json.dumps(out["metrics"], indent=2), encoding="utf-8"
    )
    (args.model_dir / "metadata.json").write_text(
        json.dumps(out["metadata"], indent=2), encoding="utf-8"
    )

    print("Saved models to:", args.model_dir)
    print("Metrics:", out["metrics"])


if __name__ == "__main__":
    main()
