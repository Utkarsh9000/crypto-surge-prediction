"""
Train surge classification and growth regression models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score


FEATURE_COLS = [
    "ret_1d",
    "ret_7d",
    "ret_30d",
    "vol_7d",
    "vol_30d",
    "vol_z",
]


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["coin_id", "date"])
    train_parts = []
    test_parts = []
    for coin_id, grp in df.groupby("coin_id"):
        cutoff = int(len(grp) * (1 - test_ratio))
        train_parts.append(grp.iloc[:cutoff])
        test_parts.append(grp.iloc[cutoff:])
    return pd.concat(train_parts), pd.concat(test_parts)


def train_models(df: pd.DataFrame) -> dict:
    train_df, test_df = time_split(df)

    x_train = train_df[FEATURE_COLS]
    y_train = train_df["surge_label"]

    x_test = test_df[FEATURE_COLS]
    y_test = test_df["surge_label"]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=7,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=7,
    )
    reg.fit(x_train, train_df["next_7d_return"])
    reg_preds = reg.predict(x_test)
    metrics["mae_return_7d"] = float(mean_absolute_error(test_df["next_7d_return"], reg_preds))

    return {
        "classifier": clf,
        "regressor": reg,
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surge models")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data)
    out = train_models(df)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(out["classifier"], args.model_dir / "surge_classifier.joblib")
    joblib.dump(out["regressor"], args.model_dir / "return_regressor.joblib")
    joblib.dump(out["metrics"], args.model_dir / "metrics.joblib")

    print("Saved models to:", args.model_dir)
    print("Metrics:", out["metrics"])


if __name__ == "__main__":
    main()
