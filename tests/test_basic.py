import pandas as pd

from cryptosurge.features import FEATURE_COLS, build_features


def _make_rows(n_days: int = 160) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    price = 100.0
    volume = 1_000_000.0
    market_cap = 10_000_000_000.0
    for i, day in enumerate(dates):
        price *= 1 + 0.001 + (i % 7) * 0.0002
        volume *= 1 + (i % 5) * 0.0003
        market_cap = price * 100_000_000
        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "coin_id": "bitcoin",
                "price": price,
                "volume": volume,
                "market_cap": market_cap,
            }
        )
    return pd.DataFrame(rows)


def test_build_features_runs():
    df = _make_rows()
    out = build_features(df)
    assert not out.empty
    assert "surge_label" in out.columns
    assert set(FEATURE_COLS).issubset(out.columns)
