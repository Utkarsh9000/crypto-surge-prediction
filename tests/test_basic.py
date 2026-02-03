import pandas as pd

from cryptosurge.features import build_features


def test_build_features_runs():
    rows = [
        {"date": "2024-01-01", "coin_id": "bitcoin", "price": 1, "volume": 10, "market_cap": 100},
        {"date": "2024-01-02", "coin_id": "bitcoin", "price": 1.1, "volume": 11, "market_cap": 101},
        {"date": "2024-01-03", "coin_id": "bitcoin", "price": 1.2, "volume": 12, "market_cap": 102},
        {"date": "2024-01-04", "coin_id": "bitcoin", "price": 1.3, "volume": 13, "market_cap": 103},
        {"date": "2024-01-05", "coin_id": "bitcoin", "price": 1.4, "volume": 14, "market_cap": 104},
        {"date": "2024-01-06", "coin_id": "bitcoin", "price": 1.5, "volume": 15, "market_cap": 105},
        {"date": "2024-01-07", "coin_id": "bitcoin", "price": 1.6, "volume": 16, "market_cap": 106},
        {"date": "2024-01-08", "coin_id": "bitcoin", "price": 1.7, "volume": 17, "market_cap": 107},
        {"date": "2024-01-09", "coin_id": "bitcoin", "price": 1.8, "volume": 18, "market_cap": 108},
        {"date": "2024-01-10", "coin_id": "bitcoin", "price": 1.9, "volume": 19, "market_cap": 109},
        {"date": "2024-01-11", "coin_id": "bitcoin", "price": 2.0, "volume": 20, "market_cap": 110},
        {"date": "2024-01-12", "coin_id": "bitcoin", "price": 2.1, "volume": 21, "market_cap": 111},
        {"date": "2024-01-13", "coin_id": "bitcoin", "price": 2.2, "volume": 22, "market_cap": 112},
        {"date": "2024-01-14", "coin_id": "bitcoin", "price": 2.3, "volume": 23, "market_cap": 113},
        {"date": "2024-01-15", "coin_id": "bitcoin", "price": 2.4, "volume": 24, "market_cap": 114},
        {"date": "2024-01-16", "coin_id": "bitcoin", "price": 2.5, "volume": 25, "market_cap": 115},
    ]
    df = pd.DataFrame(rows)
    out = build_features(df)
    assert "surge_label" in out.columns
