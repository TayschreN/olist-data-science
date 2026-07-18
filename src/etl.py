import pandas as pd
from pathlib import Path
from typing import Optional

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


def load_raw_data(path: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    path = path or DATA_RAW
    files = [
        "olist_orders_dataset",
        "olist_order_items_dataset",
        "olist_order_payments_dataset",
        "olist_order_reviews_dataset",
        "olist_customers_dataset",
        "olist_products_dataset",
        "olist_sellers_dataset",
        "product_category_name_translation",
    ]
    data = {}
    for name in files:
        filepath = path / f"{name}.csv"
        if filepath.exists():
            data[name] = pd.read_csv(filepath)
    return data


def clean_orders(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["olist_orders_dataset"].copy()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )
    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"]
    )
    orders["order_estimated_delivery_date"] = pd.to_datetime(
        orders["order_estimated_delivery_date"]
    )

    orders["delivery_delay"] = (
        orders["order_delivered_customer_date"]
        - orders["order_estimated_delivery_date"]
    ).dt.days

    return orders


def merge_datasets(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = clean_orders(data)
    items = data["olist_order_items_dataset"]
    payments = data["olist_order_payments_dataset"]
    reviews = data["olist_order_reviews_dataset"]
    customers = data["olist_customers_dataset"]
    products = data["olist_products_dataset"]
    sellers = data["olist_sellers_dataset"]
    categories = data.get("product_category_name_translation")

    df = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(items, on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(sellers, on="seller_id", how="left")
    )

    if categories is not None:
        df = df.merge(
            categories, on="product_category_name", how="left"
        )

    return df
