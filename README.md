# Crypto Surge Prediction (2024–2026) — Python-Only

This repo analyzes why crypto prices are unpredictable and builds a reproducible pipeline to **download real market data (2024–today)** for 10 major coins, engineer features, label “surge” events, and train models to forecast growth.

## Why Crypto Is Unpredictable (Short Answer)
- **High reflexivity:** narratives + leverage amplify moves.
- **Liquidity fragmentation:** spreads/volatility vary across venues.
- **Macro sensitivity:** rates and risk-on/off regimes shift fast.
- **Regulatory shocks:** binary events move prices.

## Data Source (Real Data)
We use CoinGecko’s official API to download daily price, market cap, and volume. The **range endpoint** supports time‑bounded history, but **full 2024–2026 range requires a paid or demo API key**. Demo keys only include **~1 year of daily history**; for 2024–2026 you’ll need a **Pro plan with 2+ years of history**.  
See: CoinGecko docs and pricing. citeturn0search1turn0search4

### Setup
```powershell
cd "C:\Users\utkarsh\OneDrive\Desktop\ML MODEL\crypto-surge-prediction"
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"

# API key config
$env:COINGECKO_API_KEY="your_key_here"
$env:COINGECKO_KEY_TYPE="pro"   # or "demo"
```

## Quickstart
```powershell
# 1) Download real data (2024-01-01 to today)
python -m cryptosurge.downloader --start 2024-01-01 --end today --out data/market.csv

# 2) Build features + labels
python -m cryptosurge.features --data data/market.csv --out data/features.csv

# 3) Train models
python -m cryptosurge.train --data data/features.csv --model-dir models

# 4) Predict next surge candidates
python -m cryptosurge.predict --data data/features.csv --model-dir models --out data/predictions.csv
```

## Surge Definition
By default: **surge = next‑7‑day return >= 15%**. You can change this in `features.py`.

## Structure
- `src/cryptosurge/downloader.py` fetches real CoinGecko data
- `src/cryptosurge/features.py` builds returns, volatility, momentum
- `src/cryptosurge/train.py` trains classification + regression models
- `src/cryptosurge/predict.py` generates growth forecasts

## Notes
This is a robust baseline. For production:
- add on‑chain metrics (active addresses, exchange flows)
- use regime detection and macro features
- add walk‑forward validation per coin

## License
MIT
