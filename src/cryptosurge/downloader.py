"""
Download historical market data from CoinGecko.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

from .config import DEFAULT_COINS, DEFAULT_VS_CURRENCY


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime


def _parse_date(value: str) -> datetime:
    if value.lower() == "today":
        return datetime.now(timezone.utc)
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _get_base_url(key_type: str) -> str:
    if key_type == "pro":
        return "https://pro-api.coingecko.com/api/v3"
    return "https://api.coingecko.com/api/v3"


def _get_headers(api_key: str, key_type: str) -> dict:
    if key_type == "pro":
        return {"x-cg-pro-api-key": api_key}
    return {"x-cg-demo-api-key": api_key}


def _split_range(start: datetime, end: datetime, days: int = 180) -> Iterable[DateRange]:
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=days), end)
        yield DateRange(cur, nxt)
        cur = nxt


def _fetch_range(
    coin_id: str,
    vs_currency: str,
    drange: DateRange,
    base_url: str,
    headers: dict,
    session: requests.Session,
) -> pd.DataFrame:
    url = f"{base_url}/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": vs_currency,
        "from": int(drange.start.timestamp()),
        "to": int(drange.end.timestamp()),
    }
    resp = session.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    prices = payload.get("prices", [])
    caps = payload.get("market_caps", [])
    vols = payload.get("total_volumes", [])

    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["market_cap"] = [v[1] for v in caps] if caps else None
    df["volume"] = [v[1] for v in vols] if vols else None
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
    df = df.drop(columns=["ts_ms"])
    return df


def download_market_data(
    coins: List[str],
    vs_currency: str,
    start: datetime,
    end: datetime,
    api_key: str,
    key_type: str,
) -> pd.DataFrame:
    if key_type not in {"pro", "demo"}:
        raise ValueError("key_type must be 'pro' or 'demo'")

    if key_type == "demo":
        max_days = 365
        if (end - start).days > max_days:
            raise ValueError(
                "Demo API keys only include ~1 year of history. "
                "Use a Pro key for 2024–2026 range."
            )

    base_url = _get_base_url(key_type)
    headers = _get_headers(api_key, key_type)

    frames = []
    with requests.Session() as session:
        for coin_id in coins:
            coin_frames = []
            for drange in _split_range(start, end):
                part = _fetch_range(coin_id, vs_currency, drange, base_url, headers, session)
                coin_frames.append(part)
            df = pd.concat(coin_frames, ignore_index=True)
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
            df["coin_id"] = coin_id
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download crypto market data")
    parser.add_argument("--coins", type=str, default=",".join(DEFAULT_COINS))
    parser.add_argument("--vs", type=str, default=DEFAULT_VS_CURRENCY)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="today")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    api_key = os.getenv("COINGECKO_API_KEY")
    key_type = os.getenv("COINGECKO_KEY_TYPE", "demo")
    if not api_key:
        raise RuntimeError("COINGECKO_API_KEY is not set")

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]

    df = download_market_data(coins, args.vs, start, end, api_key, key_type)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to: {args.out}")


if __name__ == "__main__":
    main()
