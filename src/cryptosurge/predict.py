"""
Generate surge probabilities and return forecasts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .features import (
    FEATURE_COLS,
    SURGE_HORIZON_DAYS,
    SURGE_THRESHOLD,
    build_features,
)


def _load_bundle(model_dir: Path) -> dict | None:
    bundle_path = model_dir / "model_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict surge probabilities")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=SURGE_HORIZON_DAYS)
    parser.add_argument("--surge-threshold", type=float, default=SURGE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)

    bundle = _load_bundle(args.model_dir)
    if bundle:
        clf = bundle["classifier"]
        reg = bundle["regressor"]
        feature_cols = bundle["metadata"]["feature_cols"]
        threshold = bundle["metrics"].get("threshold", 0.5)
        horizon = bundle["metadata"].get("horizon", args.horizon)
        surge_threshold = bundle["metadata"].get("surge_threshold", args.surge_threshold)
    else:
        clf = joblib.load(args.model_dir / "surge_classifier.joblib")
        reg = joblib.load(args.model_dir / "return_regressor.joblib")
        feature_cols = FEATURE_COLS
        threshold = 0.5
        horizon = args.horizon
        surge_threshold = args.surge_threshold

    if not set(feature_cols).issubset(df.columns):
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
            include_labels=False,
        )

    x = df[feature_cols]
    df["surge_probability"] = clf.predict_proba(x)[:, 1]
    df["surge_signal"] = (df["surge_probability"] >= threshold).astype(int)
    pred_col = f"predicted_{horizon}d_return"
    df[pred_col] = reg.predict(x)
    if horizon == 7:
        df["predicted_7d_return"] = df[pred_col]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote predictions to: {args.out}")


if __name__ == "__main__":
    main()
