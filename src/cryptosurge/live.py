"""
Live data fetch + on-the-fly training for real-time surge detection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier, DummyRegressor

from .features import (
    FEATURE_COLS,
    LABEL_COL,
    NEXT_RETURN_COL,
    SURGE_HORIZON_DAYS,
    SURGE_THRESHOLD,
    build_features,
)
from .train import train_models


@dataclass
class LiveModelBundle:
    classifier: Any
    regressor: Any
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    trained_at: datetime
    coin_id: str
    horizon: int
    surge_threshold: float


_LIVE_CACHE: dict[tuple[str, int, float], LiveModelBundle] = {}


def _get_base_url(key_type: str) -> str:
    return "https://pro-api.coingecko.com/api/v3" if key_type == "pro" else "https://api.coingecko.com/api/v3"


def _get_headers(api_key: str | None, key_type: str) -> dict:
    if not api_key:
        return {}
    if key_type == "pro":
        return {"x-cg-pro-api-key": api_key}
    return {"x-cg-demo-api-key": api_key}


def _fetch_market_chart(
    coin_id: str,
    vs_currency: str,
    days: int,
    api_key: str | None,
    key_type: str,
) -> pd.DataFrame:
    base_url = _get_base_url(key_type)
    url = f"{base_url}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    resp = requests.get(url, params=params, headers=_get_headers(api_key, key_type), timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    prices = payload.get("prices", [])
    caps = payload.get("market_caps", [])
    vols = payload.get("total_volumes", [])

    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["market_cap"] = [v[1] for v in caps] if caps else np.nan
    df["volume"] = [v[1] for v in vols] if vols else np.nan
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
    df = df.drop(columns=["ts_ms"])

    # Aggregate to daily rows (take last value per day)
    df = df.groupby("date", as_index=False).last()
    df["coin_id"] = coin_id
    return df


def _prepare_live_frames(
    coin_id: str,
    vs_currency: str,
    days: int,
    api_key: str | None,
    key_type: str,
    horizon: int,
    surge_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _fetch_market_chart(coin_id, vs_currency, days, api_key, key_type)
    if raw.empty:
        raise ValueError("No data returned from market API.")

    train_df = build_features(
        raw,
        horizon=horizon,
        surge_threshold=surge_threshold,
        include_labels=True,
    )
    pred_df = build_features(
        raw,
        horizon=horizon,
        surge_threshold=surge_threshold,
        include_labels=False,
    )
    return train_df, pred_df


def _train_or_fallback(
    df: pd.DataFrame, horizon: int, surge_threshold: float
) -> LiveModelBundle:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    y = df[LABEL_COL]

    if y.nunique() < 2:
        x = df[FEATURE_COLS]
        clf = DummyClassifier(strategy="prior")
        clf.fit(x, y)
        reg = DummyRegressor(strategy="mean")
        reg.fit(x, df[NEXT_RETURN_COL])
        metrics = {
            "auc": None,
            "pr_auc": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "brier": None,
            "precision_at_5pct": None,
            "base_rate": float(np.mean(y)) if len(y) else None,
            "threshold": 0.5,
        }
        metadata = {
            "feature_cols": FEATURE_COLS,
            "horizon": horizon,
            "surge_threshold": surge_threshold,
            "train_end": df["date"].max().isoformat() if not df.empty else None,
        }
        return LiveModelBundle(
            classifier=clf,
            regressor=reg,
            metadata=metadata,
            metrics=metrics,
            trained_at=datetime.now(timezone.utc),
            coin_id=str(df["coin_id"].iloc[-1]) if not df.empty else "unknown",
            horizon=horizon,
            surge_threshold=surge_threshold,
        )

    out = train_models(
        df,
        test_ratio=0.0,
        calib_ratio=0.2,
        horizon=horizon,
        surge_threshold=surge_threshold,
    )
    return LiveModelBundle(
        classifier=out["classifier"],
        regressor=out["regressor"],
        metadata=out["metadata"],
        metrics=out["metrics"],
        trained_at=datetime.now(timezone.utc),
        coin_id=str(df["coin_id"].iloc[-1]) if not df.empty else "unknown",
        horizon=horizon,
        surge_threshold=surge_threshold,
    )


def get_live_bundle(
    coin_id: str,
    vs_currency: str,
    days: int,
    horizon: int,
    surge_threshold: float,
    refresh: bool = False,
) -> tuple[LiveModelBundle, pd.DataFrame]:
    key = (coin_id, horizon, float(surge_threshold))
    if not refresh and key in _LIVE_CACHE:
        return _LIVE_CACHE[key], _LIVE_CACHE[key].metadata.get("pred_df", pd.DataFrame())

    api_key = os.getenv("COINGECKO_API_KEY")
    key_type = os.getenv("COINGECKO_KEY_TYPE", "demo")
    train_df, pred_df = _prepare_live_frames(
        coin_id=coin_id,
        vs_currency=vs_currency,
        days=days,
        api_key=api_key,
        key_type=key_type,
        horizon=horizon,
        surge_threshold=surge_threshold,
    )
    bundle = _train_or_fallback(train_df, horizon=horizon, surge_threshold=surge_threshold)
    bundle.metadata["pred_df"] = pred_df
    _LIVE_CACHE[key] = bundle
    return bundle, pred_df


def predict_live(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 365,
    horizon: int = 1,
    surge_threshold: float = SURGE_THRESHOLD,
    refresh: bool = False,
) -> dict[str, Any]:
    bundle, pred_df = get_live_bundle(
        coin_id=coin_id,
        vs_currency=vs_currency,
        days=days,
        horizon=horizon,
        surge_threshold=surge_threshold,
        refresh=refresh,
    )

    if pred_df.empty:
        raise ValueError("Not enough data to compute live features.")

    pred_df = pred_df.sort_values("date")
    latest = pred_df.iloc[-1]
    x = pred_df.tail(1)[bundle.metadata["feature_cols"]]

    proba = float(bundle.classifier.predict_proba(x)[:, 1][0])
    threshold = float(bundle.metrics.get("threshold", 0.5) or 0.5)
    signal = int(proba >= threshold)
    predicted_return = float(bundle.regressor.predict(x)[0])

    as_of = pd.to_datetime(latest["date"]).date().isoformat()
    target = (pd.to_datetime(latest["date"]) + pd.Timedelta(days=horizon)).date().isoformat()

    return {
        "coin_id": coin_id,
        "vs_currency": vs_currency,
        "as_of": as_of,
        "target_date": target,
        "horizon_days": horizon,
        "surge_threshold": surge_threshold,
        "surge_probability": proba,
        "surge_signal": signal,
        f"predicted_{horizon}d_return": predicted_return,
        "model": {
            "trained_at": bundle.trained_at.isoformat(),
            "train_end": bundle.metadata.get("train_end"),
            "base_rate": bundle.metrics.get("base_rate"),
        },
    }
