import pandas as pd
import numpy as np


def delivery_performance(df: pd.DataFrame) -> pd.DataFrame:
    perf = (
        df.dropna(subset=["delivery_delay"])
        .groupby("customer_state")
        .agg(
            avg_delay=("delivery_delay", "mean"),
            max_delay=("delivery_delay", "max"),
            on_time_rate=(
                "delivery_delay",
                lambda x: (x <= 0).mean() * 100,
            ),
            orders=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("on_time_rate")
    )
    return perf


def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    cat_name = (
        "product_category_name_english"
        if "product_category_name_english" in df.columns
        else "product_category_name"
    )

    perf = (
        df.groupby(cat_name)
        .agg(
            avg_review=("review_score", "mean"),
            total_orders=("order_id", "nunique"),
            total_sales=("payment_value", "sum"),
        )
        .reset_index()
        .query("total_orders >= 10")
        .sort_values("avg_review", ascending=False)
    )
    return perf


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

    trend = (
        df.drop_duplicates(subset=["order_id", "order_month"])
        .groupby("order_month")
        .agg(
            total_orders=("order_id", "nunique"),
            total_revenue=("payment_value", "sum"),
            avg_ticket=("payment_value", "mean"),
        )
        .reset_index()
        .sort_values("order_month")
    )
    return trend


def top_sellers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    sellers = (
        df.groupby(["seller_id", "seller_city", "seller_state"])
        .agg(
            total_sales=("payment_value", "sum"),
            total_orders=("order_id", "nunique"),
            avg_review=("review_score", "mean"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .head(top_n)
    )
    return sellers
