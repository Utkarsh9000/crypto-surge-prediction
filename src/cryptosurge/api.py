"""
FastAPI service for live scoring.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import FEATURE_COLS, SURGE_HORIZON_DAYS, SURGE_THRESHOLD, build_features


class Candle(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    coin_id: str
    price: float
    volume: float
    market_cap: float


class PredictRequest(BaseModel):
    rows: List[Candle]
    horizon: int | None = None
    surge_threshold: float | None = None
    top_k: float | None = Field(
        default=None, description="If set, keep only top-k probability fraction"
    )


class PredictResponse(BaseModel):
    count: int
    results: list[dict[str, Any]]


app = FastAPI(title="Crypto Surge API", version="0.1.0")

_BUNDLE: dict | None = None


def _load_bundle(model_dir: Path) -> dict:
    bundle_path = model_dir / "model_bundle.joblib"
    if bundle_path.exists():
        return joblib.load(bundle_path)
    clf_path = model_dir / "surge_classifier.joblib"
    reg_path = model_dir / "return_regressor.joblib"
    if not clf_path.exists() or not reg_path.exists():
        raise FileNotFoundError(
            "Model files not found. Expected model_bundle.joblib or "
            "surge_classifier.joblib + return_regressor.joblib."
        )
    return {
        "classifier": joblib.load(clf_path),
        "regressor": joblib.load(reg_path),
        "metadata": {"feature_cols": FEATURE_COLS, "horizon": SURGE_HORIZON_DAYS},
        "metrics": {"threshold": 0.5},
    }


def _get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    _BUNDLE = _load_bundle(model_dir)
    return _BUNDLE


def _score(df: pd.DataFrame, horizon: int, surge_threshold: float) -> pd.DataFrame:
    bundle = _get_bundle()
    feature_cols = bundle["metadata"]["feature_cols"]
    threshold = bundle.get("metrics", {}).get("threshold", 0.5)

    if not set(feature_cols).issubset(df.columns):
        required = {"date", "coin_id", "price", "volume", "market_cap"}
        if not required.issubset(df.columns):
            missing = sorted(required - set(df.columns))
            raise ValueError("Missing required columns: " + ", ".join(missing))
        df = build_features(
            df,
            horizon=horizon,
            surge_threshold=surge_threshold,
            include_labels=False,
        )

    if df.empty:
        raise ValueError("Not enough history to build features.")

    x = df[feature_cols]
    df = df.copy()
    df["surge_probability"] = bundle["classifier"].predict_proba(x)[:, 1]
    df["surge_signal"] = (df["surge_probability"] >= threshold).astype(int)
    df[f"predicted_{horizon}d_return"] = bundle["regressor"].predict(x)
    return df


def _to_results(df: pd.DataFrame, horizon: int) -> list[dict[str, Any]]:
    cols = [
        "date",
        "coin_id",
        "surge_probability",
        "surge_signal",
        f"predicted_{horizon}d_return",
    ]
    out = df[cols].copy()
    out["date"] = out["date"].astype(str)
    return out.to_dict(orient="records")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")

    df = pd.DataFrame([row.model_dump() for row in req.rows])
    horizon = req.horizon or SURGE_HORIZON_DAYS
    surge_threshold = req.surge_threshold or SURGE_THRESHOLD

    try:
        scored = _score(df, horizon=horizon, surge_threshold=surge_threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.top_k:
        k = max(1, int(len(scored) * req.top_k))
        scored = scored.sort_values("surge_probability", ascending=False).head(k)

    results = _to_results(scored, horizon=horizon)
    return PredictResponse(count=len(results), results=results)


@app.post("/alerts", response_model=PredictResponse)
def alerts(req: PredictRequest) -> PredictResponse:
    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")

    df = pd.DataFrame([row.model_dump() for row in req.rows])
    horizon = req.horizon or SURGE_HORIZON_DAYS
    surge_threshold = req.surge_threshold or SURGE_THRESHOLD

    try:
        scored = _score(df, horizon=horizon, surge_threshold=surge_threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scored = scored[scored["surge_signal"] == 1]
    if req.top_k:
        k = max(1, int(len(scored) * req.top_k))
        scored = scored.sort_values("surge_probability", ascending=False).head(k)

    results = _to_results(scored, horizon=horizon)
    return PredictResponse(count=len(results), results=results)


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("cryptosurge.api:app", host=host, port=port, reload=False)
