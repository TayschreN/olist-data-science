from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender import recommend_for_customer, recommend_by_category

app = FastAPI(title="Olist Recommender API", version="0.1.0")

DATA_PATH = Path("data/processed/olist_full.parquet")
MODELS_DIR = Path("models")

df = None


@app.on_event("startup")
def load_artifacts():
    global df
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
    else:
        print("Aviso: dados não encontrados. Faça o pipeline primeiro.")


@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": df is not None}


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str, top_n: int = 5):
    if df is None:
        raise HTTPException(400, "Dados não carregados")

    if customer_id not in df["customer_unique_id"].values:
        raise HTTPException(404, "Cliente não encontrado")

    recs = recommend_for_customer(df, customer_id, top_n=top_n)
    products = []
    for pid, score in recs:
        product_info = df[df["product_id"] == pid][
            ["product_category_name_english", "product_weight_g"]
        ].iloc[0]
        products.append({
            "product_id": pid,
            "category": product_info["product_category_name_english"],
            "score": score,
        })

    return {"customer_id": customer_id, "recommendations": products}


@app.get("/similar/{product_id}")
def similar_products(product_id: str, top_n: int = 5):
    if df is None:
        raise HTTPException(400, "Dados não carregados")

    recs = recommend_by_category(df, product_id, top_n=top_n)
    return {"product_id": product_id, "similar": recs}


@app.get("/customer/{customer_id}")
def customer_info(customer_id: str):
    if df is None:
        raise HTTPException(400, "Dados não carregados")

    customer_data = df[df["customer_unique_id"] == customer_id]

    if customer_data.empty:
        raise HTTPException(404, "Cliente não encontrado")

    return {
        "customer_id": customer_id,
        "city": customer_data["customer_city"].iloc[0],
        "state": customer_data["customer_state"].iloc[0],
        "total_orders": customer_data["order_id"].nunique(),
        "total_spent": float(customer_data["payment_value"].sum()),
        "avg_review": float(customer_data["review_score"].mean()),
    }
