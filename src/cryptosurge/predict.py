"""
Generate surge probabilities and next-7-day return forecasts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .train import FEATURE_COLS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict surge probabilities")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)

    clf = joblib.load(args.model_dir / "surge_classifier.joblib")
    reg = joblib.load(args.model_dir / "return_regressor.joblib")

    x = df[FEATURE_COLS]
    df["surge_probability"] = clf.predict_proba(x)[:, 1]
    df["predicted_7d_return"] = reg.predict(x)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote predictions to: {args.out}")


if __name__ == "__main__":
    main()
