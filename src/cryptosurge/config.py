from __future__ import annotations

from dataclasses import dataclass


DEFAULT_COINS = [
    "bitcoin",
    "ethereum",
    "tether",
    "binancecoin",
    "solana",
    "ripple",
    "usd-coin",
    "cardano",
    "dogecoin",
    "tron",
]


@dataclass(frozen=True)
class ApiConfig:
    api_key: str
    key_type: str  # "pro" or "demo"


DEFAULT_VS_CURRENCY = "usd"
