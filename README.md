# Crypto Surge Prediction (2024-2026)

This project downloads real crypto market data, builds features, labels surge events, and trains models that output both a surge probability and a return forecast. It also includes a deployable FastAPI service and walk-forward backtesting.

## Why Surges Are Hard
- High reflexivity: narratives + leverage amplify moves
- Liquidity fragmentation: spreads and volatility vary across venues
- Macro sensitivity: risk-on and risk-off regimes shift fast
- Regulatory shocks: binary events move prices

## Data Source
We use CoinGecko API data for daily price, market cap, and volume. Demo keys provide about one year of history; Pro keys provide longer ranges.

### Setup
```powershell
cd "C:\Users\utkarsh\OneDrive\Desktop\ML MODEL\crypto-surge-prediction"
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"

# API key config
$env:COINGECKO_API_KEY="your_key_here"
$env:COINGECKO_KEY_TYPE="pro"   # or "demo"
```

For tests:
```powershell
python -m pip install -r requirements-dev.txt
```

## Quickstart
```powershell
# 1) Download market data
python -m cryptosurge.downloader --start 2024-01-01 --end today --out data/market.csv

# 2) Build features + labels
python -m cryptosurge.features --data data/market.csv --out data/features.csv --horizon 7 --surge-threshold 0.15

# 3) Train models (time-based split + calibration)
python -m cryptosurge.train --data data/features.csv --model-dir models --test-ratio 0.2 --calib-ratio 0.2

# 4) Predict
python -m cryptosurge.predict --data data/features.csv --model-dir models --out data/predictions.csv

# 5) Evaluate on a time window
python -m cryptosurge.evaluate --data data/features.csv --model-dir models --start 2025-01-01 --out models/metrics_eval.json
```

## Walk-Forward Backtest
```powershell
python -m cryptosurge.walkforward --data data/features.csv --out data/walkforward.csv --train-days 365 --test-days 30 --step-days 30
```

## Live API
Run the service locally:
```powershell
$env:MODEL_DIR="models"
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API expects enough history to compute rolling features (recommend 90+ daily rows per coin).

Example request:
```json
{
  "rows": [
    {"date": "2026-01-01", "coin_id": "bitcoin", "price": 42000, "volume": 22000000000, "market_cap": 825000000000},
    {"date": "2026-01-02", "coin_id": "bitcoin", "price": 43200, "volume": 21000000000, "market_cap": 840000000000}
  ],
  "top_k": 0.05
}
```

Endpoints:
- `GET /health`
- `POST /predict` returns probabilities and predicted returns
- `POST /alerts` returns only rows above the surge threshold

## Deploy
### Vercel
This repo is Vercel-ready. Vercel supports FastAPI with zero configuration; it auto-detects Python Serverless Functions when you add an `api/` directory at the project root. We provide `api/index.py` that re-exports the FastAPI `app` from `app.py`, so Vercel can deploy it directly.

The Python runtime respects `requires-python` and reads dependencies from `pyproject.toml` or `requirements.txt`, which we already provide.

Note: On Vercel, the API endpoints are available under `/api` (for example `/api/health`, `/api/predict`, `/api/alerts`).

### Docker
```powershell
# build
docker build -t cryptosurge .

# run
docker run -p 8000:8000 -e MODEL_DIR=models cryptosurge
```

If you want the image to contain trained model artifacts, remove `models/` and `*.joblib` from `.dockerignore` or mount them at runtime and set `MODEL_DIR`.

### Procfile Platforms
A `Procfile` is included for platforms that support it.

## Model Artifacts
By default, `.gitignore` excludes `models/` and `*.joblib`. If you want to deploy from GitHub and ship a trained model, you can either:
- remove `models/` and `*.joblib` from `.gitignore`, then commit the artifacts, or
- store artifacts elsewhere and set `MODEL_DIR` at deploy time.

## Surge Definition
By default: surge = next-7-day return >= 15%. Override with `--surge-threshold` or update `SURGE_THRESHOLD` in `features.py`.

## What Was Upgraded
- Expanded feature set: momentum, volatility, RSI, MACD, Bollinger, market signals, BTC dominance
- Time-based train/test splits to reduce leakage
- Probability calibration and threshold tuning
- Richer metrics: PR-AUC, precision@k, and signal hit rate

## Notes On "99% Accuracy"
Accuracy is not a good target for rare events. It can be very high even if the model never predicts a surge. Use PR-AUC, precision@k, or expected return instead.

## Structure
- `src/cryptosurge/downloader.py` fetches CoinGecko data
- `src/cryptosurge/features.py` builds features and labels
- `src/cryptosurge/train.py` trains classification + regression models
- `src/cryptosurge/predict.py` generates forecasts
- `src/cryptosurge/evaluate.py` scores performance on labeled data
- `src/cryptosurge/api.py` serves the live API
- `src/cryptosurge/walkforward.py` runs walk-forward backtests

## License
MIT
