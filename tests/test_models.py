import pandas as pd
import numpy as np
from src.models import segment_customers, elbow_method


def test_segment_customers():
    rfv = pd.DataFrame({
        "recency": [1, 10, 30, 60, 5, 20, 40, 80],
        "frequency": [10, 5, 2, 1, 8, 3, 2, 1],
        "monetary": [1000, 500, 200, 50, 800, 300, 150, 30],
    })

    result, model, scaler = segment_customers(rfv, n_clusters=3, save=False)

    assert "cluster" in result.columns
    assert result["cluster"].nunique() == 3
    assert hasattr(model, "cluster_centers_")


def test_elbow_method():
    rfv = pd.DataFrame({
        "recency": np.random.randint(1, 100, 50),
        "frequency": np.random.randint(1, 10, 50),
        "monetary": np.random.uniform(50, 1000, 50),
    })

    inertias = elbow_method(rfv, max_k=5)
    assert len(inertias) == 5
    assert all(isinstance(i, float) for i in inertias)
