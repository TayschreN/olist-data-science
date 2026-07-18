import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import delivery_performance, monthly_sales_trend, category_performance
from src.recommender import recommend_for_customer
from src.features import compute_rfv, rfm_score
from src.etl import load_raw_data, merge_datasets

st.set_page_config(
    page_title="Olist Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)

DATA_PATH = Path("data/processed/olist_full.parquet")
RFV_PATH = Path("data/processed/rfv_clusters.csv")


@st.cache_data
def load_full_data():
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    raw = load_raw_data()
    if raw:
        return merge_datasets(raw)
    return None


@st.cache_data
def load_rfv_data():
    if RFV_PATH.exists():
        return pd.read_csv(RFV_PATH)
    df = load_full_data()
    if df is not None:
        rfv = compute_rfv(df)
        return rfm_score(rfv)
    return None


df = load_full_data()
rfv = load_rfv_data()

full_mode = df is not None

st.title(":bar_chart: Olist E-commerce Analytics")
st.markdown("---")

if not full_mode:
    st.info(
        "**Modo demonstração**: Dados completos não disponíveis. "
        "Para a experiência completa, execute o pipeline localmente."
    )

col1, col2, col3, col4 = st.columns(4)

if full_mode:
    total_revenue = df["payment_value"].sum()
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_unique_id"].nunique()
    avg_review = df["review_score"].mean()
else:
    total_revenue = 0
    total_orders = 0
    total_customers = len(rfv) if rfv is not None else 0
    avg_review = 0

col1.metric("Receita Total", f"R$ {total_revenue:,.0f}" if full_mode else "N/A")
col2.metric("Pedidos", f"{total_orders:,}" if full_mode else "N/A")
col3.metric("Clientes", f"{total_customers:,}")
col4.metric("Review Médio", f"{avg_review:.2f}" if full_mode else "N/A")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    [":chart_with_upwards_trend: Tendências", ":busts_in_silhouette: Clientes", ":package: Produtos", ":dart: Recomendação"]
)

with tab1:
    if full_mode:
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
    else:
        st.info("Carregue os dados completos para ver tendências e performance.")

with tab2:
    if rfv is not None and "cluster" in rfv.columns:
        st.subheader("Segmentação RFV")

        cluster_stats = (
            rfv.groupby("cluster")[["recency", "frequency", "monetary"]]
            .mean()
            .round(1)
        )

        cluster_labels = {
            0: "Clientes Regulares",
            1: "VIP",
            2: "Clientes Novatos",
            3: "Em Risco",
        }
        cluster_stats = cluster_stats.rename(index=cluster_labels)
        st.dataframe(cluster_stats, use_container_width=True)

        rfv_for_plot = rfv.copy()
        rfv_for_plot["segment"] = rfv_for_plot["cluster"].map(cluster_labels)

        fig3 = px.scatter(
            rfv_for_plot,
            x="recency",
            y="monetary",
            color="segment",
            size="frequency",
            hover_data=["customer_id"],
            title="Clientes por Recência e Valor (cor = segmento)",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Top Clientes (RFV Score)")
        top_rfm = rfv.sort_values("rfm_score", ascending=False).head(10)
        st.dataframe(top_rfm, use_container_width=True)

        st.subheader("Distribuição dos Segmentos")
        seg_counts = rfv_for_plot["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segmento", "Clientes"]
        fig_pie = px.pie(seg_counts, values="Clientes", names="Segmento", title="Proporção de Clientes por Segmento")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Dados RFV não disponíveis.")

with tab3:
    if full_mode:
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
    else:
        st.info("Carregue os dados completos para ver performance por categoria.")

with tab4:
    st.subheader("Sistema de Recomendação")
    if full_mode:
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
    else:
        st.info("Carregue os dados completos para usar o sistema de recomendação.")
