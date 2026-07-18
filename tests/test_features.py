import pandas as pd
import numpy as np
from datetime import datetime
from src.features import compute_rfv, rfm_score


def test_compute_rfv():
    df = pd.DataFrame({
        "customer_unique_id": ["a", "a", "b"],
        "order_purchase_timestamp": pd.to_datetime(
            ["2024-01-01", "2024-02-01", "2024-01-15"]
        ),
        "order_id": ["o1", "o2", "o3"],
        "payment_value": [100.0, 200.0, 150.0],
    })

    rfv = compute_rfv(df)

    assert "recency" in rfv.columns
    assert "frequency" in rfv.columns
    assert "monetary" in rfv.columns
    assert len(rfv) == 2


def test_rfm_score():
    rfv = pd.DataFrame({
        "customer_id": ["a", "b", "c", "d"],
        "recency": [1, 10, 30, 60],
        "frequency": [10, 5, 2, 1],
        "monetary": [1000, 500, 200, 50],
    })

    scored = rfm_score(rfv)
    assert "rfm_score" in scored.columns
    assert "r_quartile" in scored.columns
    assert scored["rfm_score"].between(3, 12).all()
