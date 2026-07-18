import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import delivery_performance, monthly_sales_trend, category_performance
from src.recommender import recommend_for_customer

st.set_page_config(
    page_title="Olist Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = Path("data/processed/olist_full.parquet")
RFV_PATH = Path("data/processed/rfv_clusters.csv")


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    rfv = pd.read_csv(RFV_PATH)
    return df, rfv


st.title("📊 Olist E-commerce Analytics")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

df, rfv = load_data()

total_revenue = df["payment_value"].sum()
total_orders = df["order_id"].nunique()
total_customers = df["customer_unique_id"].nunique()
avg_review = df["review_score"].mean()

col1.metric("Receita Total", f"R$ {total_revenue:,.0f}")
col2.metric("Pedidos", f"{total_orders:,}")
col3.metric("Clientes", f"{total_customers:,}")
col4.metric("Review Médio", f"{avg_review:.2f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Tendências", "👥 Clientes", "📦 Produtos", "🎯 Recomendação"]
)

with tab1:
    st.subheader("Receita e Pedidos por Mês")
    trend = monthly_sales_trend(df)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["order_month"],
            y=trend["total_revenue"],
            mode="lines+markers",
            name="Receita (R$)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=trend["order_month"],
            y=trend["total_orders"],
            name="Pedidos",
            yaxis="y2",
            opacity=0.3,
        )
    )
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance por Estado")
    delivery = delivery_performance(df)
    fig2 = px.bar(
        delivery.sort_values("on_time_rate"),
        x="customer_state",
        y="on_time_rate",
        color="on_time_rate",
        color_continuous_scale="RdYlGn",
        title="Taxa de Entrega no Prazo por Estado",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Segmentação RFV")

    cluster_stats = (
        rfv.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .round(1)
    )
    st.dataframe(cluster_stats, use_container_width=True)

    fig3 = px.scatter(
        rfv,
        x="recency",
        y="monetary",
        color="cluster",
        size="frequency",
        hover_data=["customer_id"],
        title="Clientes por Recência e Valor (cor = cluster)",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top Clientes (RFV Score)")
    top_rfm = rfv.sort_values("rfm_score", ascending=False).head(10)
    st.dataframe(top_rfm, use_container_width=True)

with tab3:
    st.subheader("Performance por Categoria")
    cat_perf = category_performance(df)
    fig4 = px.bar(
        cat_perf.head(20),
        x="avg_review",
        y="product_category_name_english",
        color="total_orders",
        orientation="h",
        title="Review Médio por Categoria",
        color_continuous_scale="Blues",
    )
    fig4.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(cat_perf.head(20), use_container_width=True)

with tab4:
    st.subheader("Sistema de Recomendação")
    customer_input = st.text_input("ID do Cliente", value="")

    if customer_input:
        recs = recommend_for_customer(df, customer_input)
        if recs:
            st.success(f"Recomendações para {customer_input}:")
            rec_data = []
            for pid, score in recs:
                info = df[df["product_id"] == pid][
                    ["product_category_name_english"]
                ].iloc[0]
                rec_data.append(
                    {
                        "Product ID": pid,
                        "Categoria": info["product_category_name_english"],
                        "Score": score,
                    }
                )
            st.dataframe(pd.DataFrame(rec_data), use_container_width=True)
        else:
            st.warning("Nenhuma recomendação encontrada ou cliente não existe.")
