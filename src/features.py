import pandas as pd
import numpy as np
from datetime import datetime


def compute_rfv(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfv = (
        df.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (ref_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum"),
        )
        .reset_index()
    )

    rfv.columns = ["customer_id", "recency", "frequency", "monetary"]

    for col in ["recency", "frequency", "monetary"]:
        rfv[col] = rfv[col].clip(lower=rfv[col].quantile(0.01), upper=rfv[col].quantile(0.99))

    return rfv


def rfm_score(rfv: pd.DataFrame) -> pd.DataFrame:
    df = rfv.copy()

    df["r_quartile"] = pd.qcut(df["recency"], 4, labels=[4, 3, 2, 1])
    df["f_quartile"] = pd.qcut(df["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    df["m_quartile"] = pd.qcut(df["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4])

    df["rfm_score"] = (
        df["r_quartile"].astype(int)
        + df["f_quartile"].astype(int)
        + df["m_quartile"].astype(int)
    )

    return df
