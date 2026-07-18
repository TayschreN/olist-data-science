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
            total_customers=("customer_unique_id", "nunique"),
            avg_price=("price", "mean"),
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


def order_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates(subset=["order_id"])
        .groupby("order_status")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def payment_method_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("payment_type")
        .agg(
            total=("payment_value", "sum"),
            count=("order_id", "nunique"),
            avg=("payment_value", "mean"),
        )
        .reset_index()
        .sort_values("total", ascending=False)
    )


def review_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("review_score")
        .size()
        .reset_index(name="count")
        .assign(pct=lambda x: x["count"] / x["count"].sum() * 100)
    )


def orders_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["order_id"]).copy()
    df["order_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["order_hour"] = df["order_purchase_timestamp"].dt.hour

    days = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    heatmap = (
        df.groupby(["order_dayofweek", "order_hour"])
        .size()
        .reset_index(name="count")
    )
    heatmap["day_name"] = heatmap["order_dayofweek"].map(lambda x: days[x])
    return heatmap


def payment_installments_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("payment_installments")
        .agg(
            count=("order_id", "nunique"),
            total=("payment_value", "sum"),
            avg_ticket=("payment_value", "mean"),
        )
        .reset_index()
        .query("payment_installments > 0")
    )


def delivery_delay_distribution(df: pd.DataFrame) -> pd.DataFrame:
    delays = df.dropna(subset=["delivery_delay"]).copy()
    delays["delay_category"] = pd.cut(
        delays["delivery_delay"],
        bins=[-999, -10, -1, 0, 1, 5, 10, 30, 100, 999],
        labels=[
            "Muito adiantado",
            "Adiantado",
            "No prazo",
            "1 dia atraso",
            "2-5 dias",
            "6-10 dias",
            "11-30 dias",
            "31-100 dias",
            "Muito atrasado",
        ],
    )
    return (
        delays.groupby("delay_category", observed=True)
        .size()
        .reset_index(name="count")
    )
