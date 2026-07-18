import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

MODELS_DIR = Path("models")


def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="customer_unique_id",
        columns="product_id",
        values="review_score",
        aggfunc="mean",
    ).fillna(0)


def recommend_by_category(
    df: pd.DataFrame, product_id: str, top_n: int = 5
) -> list[str]:
    product_cat = df.loc[
        df["product_id"] == product_id, "product_category_name_english"
    ].iloc[0] if not df[df["product_id"] == product_id].empty else None

    if product_cat is None:
        return []

    same_category = df[
        (df["product_category_name_english"] == product_cat)
        & (df["product_id"] != product_id)
    ]

    top_products = (
        same_category.groupby("product_id")
        .agg(avg_score=("review_score", "mean"), count=("review_score", "count"))
        .query("count >= 10")
        .sort_values("avg_score", ascending=False)
        .head(top_n)
        .index.tolist()
    )

    return top_products


def recommend_for_customer(
    df: pd.DataFrame,
    customer_id: str,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    customer_products = df[df["customer_unique_id"] == customer_id][
        "product_id"
    ].unique()

    product_scores = {}
    for prod in customer_products:
        similar = recommend_by_category(df, prod, top_n=top_n)
        for rec in similar:
            product_scores[rec] = product_scores.get(rec, 0) + 1

    customer_categories = df[df["customer_unique_id"] == customer_id][
        "product_category_name_english"
    ].unique()

    popular_in_cats = (
        df[df["product_category_name_english"].isin(customer_categories)]
        .groupby("product_id")
        .agg(avg_score=("review_score", "mean"), count=("review_score", "count"))
        .query("count >= 10")
        .sort_values("avg_score", ascending=False)
        .head(top_n)
    )

    for pid in popular_in_cats.index:
        if pid not in product_scores and pid not in customer_products:
            product_scores[pid] = 1

    sorted_recs = sorted(
        product_scores.items(), key=lambda x: x[1], reverse=True
    )

    return sorted_recs[:top_n]
