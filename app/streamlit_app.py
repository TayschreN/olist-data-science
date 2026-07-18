import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import (
    delivery_performance,
    monthly_sales_trend,
    category_performance,
    order_status_distribution,
    payment_method_breakdown,
    review_distribution,
    orders_heatmap,
    payment_installments_analysis,
    delivery_delay_distribution,
)
from src.recommender import recommend_for_customer, recommend_by_category
from src.features import compute_rfv, rfm_score
from src.etl import load_raw_data, merge_datasets

st.set_page_config(
    page_title="Olist Analytics",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/processed/olist_full.parquet")
RFV_PATH = Path("data/processed/rfv_clusters.csv")

cluster_labels = {
    0: "Regulares",
    1: "VIP",
    2: "Novatos",
    3: "Em Risco",
}
cluster_colors = {
    "VIP": "#FFD700",
    "Regulares": "#2E86AB",
    "Novatos": "#A23B72",
    "Em Risco": "#F18F01",
}


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

if rfv is not None and "cluster" in rfv.columns:
    rfv["segment"] = rfv["cluster"].map(cluster_labels)

with st.sidebar:
    st.image("https://raw.githubusercontent.com/olist/olist-ux/master/logo/logo.png", width=200)
    st.markdown("## Filtros")
    st.markdown("---")

    if full_mode:
        estados = sorted(df["customer_state"].dropna().unique())
        estados_sel = st.multiselect(
            "Estado", estados, default=[], placeholder="Selecionar estados..."
        )

        categorias = sorted(
            df["product_category_name_english"].dropna().unique()
        )
        cat_sel = st.multiselect(
            "Categoria", categorias, default=[], placeholder="Selecionar categorias..."
        )

        status_opts = sorted(df["order_status"].dropna().unique())
        status_sel = st.multiselect(
            "Status do Pedido", status_opts, default=[], placeholder="Selecionar status..."
        )

        st.markdown("---")

    if rfv is not None:
        seg_opts = list(cluster_labels.values())
        seg_sel = st.multiselect(
            "Segmento de Cliente", seg_opts, default=seg_opts,
        )

    st.markdown("---")
    st.markdown(
        "**Dataset:** [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)"
    )

df_filtered = df.copy() if full_mode else None
if full_mode and estados_sel:
    df_filtered = df_filtered[df_filtered["customer_state"].isin(estados_sel)]
if full_mode and cat_sel:
    df_filtered = df_filtered[df_filtered["product_category_name_english"].isin(cat_sel)]
if full_mode and status_sel:
    df_filtered = df_filtered[df_filtered["order_status"].isin(status_sel)]

rfv_filtered = rfv.copy() if rfv is not None else None
if rfv is not None and seg_sel:
    rfv_filtered = rfv_filtered[rfv_filtered["segment"].isin(seg_sel)]

st.title(":bar_chart: Olist E-commerce Analytics")
st.markdown(
    "*Análise completa de performance, clientes e produtos — "
    "segmentação RFV + sistema de recomendação*"
)

if not full_mode:
    st.info(
        ":information_source: **Modo demonstração**: Dados completos não disponíveis. "
        "Execute `python scripts/run_pipeline.py` para ativar todas as funcionalidades."
    )

with st.container():
    st.subheader(":bell: Resumo Executivo")
    if full_mode:
        total_rev = df_filtered["payment_value"].sum()
        total_ord = df_filtered["order_id"].nunique()
        total_cli = df_filtered["customer_unique_id"].nunique()
        avg_tkt = df_filtered["payment_value"].mean()
        avg_delay = df_filtered["delivery_delay"].mean()
        on_time = (df_filtered["delivery_delay"].dropna() <= 0).mean() * 100
        cancel_rate = (
            df_filtered[df_filtered["order_status"] == "canceled"]["order_id"].nunique()
            / total_ord * 100
        )
        avg_rev = df_filtered["review_score"].mean()
    else:
        total_rev = total_ord = total_cli = avg_tkt = avg_delay = avg_rev = 0
        on_time = cancel_rate = 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Receita Total", f"R$ {total_rev:,.0f}" if full_mode else "N/A")
    k2.metric("Pedidos", f"{total_ord:,}" if full_mode else "N/A")
    k3.metric("Clientes Únicos", f"{total_cli:,}" if rfv_filtered is not None else "N/A")
    k4.metric("Ticket Médio", f"R$ {avg_tkt:,.2f}" if full_mode else "N/A")
    k5.metric("Review Médio", f"{avg_rev:.2f}" if full_mode else "N/A")
    k6.metric("Entrega no Prazo", f"{on_time:.1f}%" if full_mode else "N/A")

    if full_mode:
        st.markdown(
            f":bulb: **Insight:** A taxa de entrega no prazo é de **{on_time:.1f}%** "
            f"com atraso médio de **{avg_delay:.1f} dias**. "
            f"A taxa de cancelamento é de **{cancel_rate:.1f}%**. "
            f"O ticket médio dos pedidos é **R$ {avg_tkt:,.2f}** "
            f"com review score médio de **{avg_rev:.2f}/5**."
        )

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":chart_with_upwards_trend: Visão Geral",
    ":calendar: Tendências",
    ":busts_in_silhouette: Clientes",
    ":package: Produtos",
    ":dart: Recomendação",
])

# =============================================================================
# TAB 1 - VISÃO GERAL
# =============================================================================
with tab1:
    if full_mode:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Status dos Pedidos")
            status_df = order_status_distribution(df_filtered)
            fig_status = px.pie(
                status_df,
                values="count",
                names="order_status",
                title="Distribuição por Status",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_status, use_container_width=True)

            st.subheader("Métodos de Pagamento")
            pay_df = payment_method_breakdown(df_filtered)
            fig_pay = px.bar(
                pay_df,
                x="payment_type",
                y="total",
                color="payment_type",
                text_auto=".1s",
                title="Volume por Método de Pagamento (R$)",
            )
            fig_pay.update_layout(showlegend=False)
            st.plotly_chart(fig_pay, use_container_width=True)

        with c2:
            st.subheader("Distribuição das Notas de Review")
            rev_df = review_distribution(df_filtered)
            fig_rev = px.bar(
                rev_df,
                x="review_score",
                y="pct",
                text_auto=".1f",
                title="% de Reviews por Nota",
                color="review_score",
                color_continuous_scale="RdYlGn",
                range_color=[1, 5],
            )
            fig_rev.update_layout(xaxis=dict(tickmode="linear"), showlegend=False)
            st.plotly_chart(fig_rev, use_container_width=True)

            st.subheader("Parcelamento")
            inst_df = payment_installments_analysis(df_filtered)
            inst_df = inst_df[inst_df["payment_installments"] <= 12]
            fig_inst = px.bar(
                inst_df,
                x="payment_installments",
                y="count",
                title="Pedidos por Nº de Parcelas",
                color="count",
                color_continuous_scale="Blues",
            )
            fig_inst.update_layout(showlegend=False)
            st.plotly_chart(fig_inst, use_container_width=True)

        st.subheader("Análise de Atrasos nas Entregas")
        delay_df = delivery_delay_distribution(df_filtered)
        fig_delay = px.bar(
            delay_df,
            x="delay_category",
            y="count",
            color="count",
            title="Distribuição dos Atrasos nas Entregas",
            color_continuous_scale="RdYlGn_r",
            text_auto=True,
        )
        fig_delay.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_delay, use_container_width=True)

        if on_time < 80:
            st.warning(
                f":warning: **Alerta:** Apenas **{on_time:.1f}%** dos pedidos foram entregues no prazo. "
                f"Isso representa um risco para a satisfação dos clientes. "
                f"Considere revisar a logística nos estados com pior performance."
            )
        else:
            st.success(
                f":white_check_mark: **Boa performance:** **{on_time:.1f}%** dos pedidos entregues no prazo. "
                f"O atraso médio é de apenas **{avg_delay:.1f} dias**."
            )
    else:
        st.info("Carregue os dados completos para ver a visão geral.")

# =============================================================================
# TAB 2 - TENDÊNCIAS
# =============================================================================
with tab2:
    if full_mode:
        trend = monthly_sales_trend(df_filtered)

        st.subheader("Receita e Pedidos por Mês")
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Scatter(
                x=trend["order_month"],
                y=trend["total_revenue"],
                mode="lines+markers",
                name="Receita (R$)",
                line=dict(color="#2E86AB", width=3),
                marker=dict(size=8),
            ),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Bar(
                x=trend["order_month"],
                y=trend["total_orders"],
                name="Pedidos",
                opacity=0.3,
                marker_color="#F18F01",
            ),
            secondary_y=True,
        )
        fig_trend.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_trend.update_yaxes(title_text="Receita (R$)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Pedidos", secondary_y=True)
        st.plotly_chart(fig_trend, use_container_width=True)

        rev_growth = (
            (trend["total_revenue"].iloc[-1] - trend["total_revenue"].iloc[0])
            / trend["total_revenue"].iloc[0] * 100
        )
        ord_growth = (
            (trend["total_orders"].iloc[-1] - trend["total_orders"].iloc[0])
            / trend["total_orders"].iloc[0] * 100
        )
        st.info(
            f":bulb: **Insight:** Do primeiro ao último mês, a receita "
            f"{'cresceu' if rev_growth > 0 else 'caiu'} **{abs(rev_growth):.1f}%** "
            f"e o volume de pedidos {'cresceu' if ord_growth > 0 else 'caiu'} **{abs(ord_growth):.1f}%**."
        )

        st.subheader("Ticket Médio por Mês")
        fig_ticket = px.line(
            trend,
            x="order_month",
            y="avg_ticket",
            markers=True,
            title="Evolução do Ticket Médio (R$)",
        )
        fig_ticket.update_traces(line=dict(color="#A23B72", width=3))
        fig_ticket.update_layout(yaxis_title="Ticket Médio (R$)")
        st.plotly_chart(fig_ticket, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Heatmap: Pedidos por Dia da Semana e Hora")
            heat = orders_heatmap(df_filtered)
            fig_heat = px.density_heatmap(
                heat,
                x="order_hour",
                y="day_name",
                z="count",
                title="Volume de Pedidos",
                color_continuous_scale="Viridis",
                nbinsx=24,
            )
            fig_heat.update_layout(
                xaxis_title="Hora do Dia",
                yaxis_title="Dia da Semana",
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
                ),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with c2:
            st.subheader("Performance de Entrega por Estado")
            delivery = delivery_performance(df_filtered)
            fig_del = px.scatter(
                delivery,
                x="on_time_rate",
                y="avg_delay",
                size="orders",
                color="customer_state",
                hover_name="customer_state",
                title="Entrega: Taxa no Prazo vs Atraso Médio",
                labels={
                    "on_time_rate": "Taxa no Prazo (%)",
                    "avg_delay": "Atraso Médio (dias)",
                },
            )
            fig_del.add_hline(
                y=delivery["avg_delay"].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Média: {delivery['avg_delay'].mean():.1f}d",
            )
            fig_del.add_vline(
                x=delivery["on_time_rate"].mean(),
                line_dash="dash",
                line_color="green",
                annotation_text=f"Média: {delivery['on_time_rate'].mean():.1f}%",
            )
            st.plotly_chart(fig_del, use_container_width=True)

        st.subheader("Top 10 Estados por Taxa no Prazo")
        c1, c2 = st.columns(2)
        with c1:
            best = delivery.sort_values("on_time_rate", ascending=False).head(10)
            fig_best = px.bar(
                best,
                x="on_time_rate",
                y="customer_state",
                orientation="h",
                color="on_time_rate",
                color_continuous_scale="Greens",
                title="Melhores Estados",
                text_auto=".1f",
            )
            fig_best.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_best, use_container_width=True)
        with c2:
            worst = delivery.sort_values("on_time_rate").head(10)
            fig_worst = px.bar(
                worst,
                x="on_time_rate",
                y="customer_state",
                orientation="h",
                color="on_time_rate",
                color_continuous_scale="Reds",
                title="Piores Estados",
                text_auto=".1f",
            )
            fig_worst.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_worst, use_container_width=True)
    else:
        st.info("Carregue os dados completos para ver tendências.")

# =============================================================================
# TAB 3 - CLIENTES
# =============================================================================
with tab3:
    if rfv_filtered is not None and "cluster" in rfv_filtered.columns:
        c1, c2 = st.columns([1, 1.5])

        with c1:
            st.subheader("Perfil dos Segmentos")
            cluster_stats = (
                rfv_filtered.groupby("segment")[["recency", "frequency", "monetary"]]
                .agg(["mean", "median", "std"])
                .round(1)
            )
            cluster_stats.columns = ["_".join(c) for c in cluster_stats.columns]
            cluster_stats = cluster_stats.reset_index()
            st.dataframe(cluster_stats, use_container_width=True)

        with c2:
            st.subheader("Distribuição dos Segmentos")
            seg_counts = rfv_filtered["segment"].value_counts().reset_index()
            seg_counts.columns = ["Segmento", "Clientes"]
            fig_seg = px.pie(
                seg_counts,
                values="Clientes",
                names="Segmento",
                hole=0.4,
                color="Segmento",
                color_discrete_map=cluster_colors,
            )
            fig_seg.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        total_seg = len(rfv_filtered)
        if "VIP" in rfv_filtered["segment"].values:
            vip_count = len(rfv_filtered[rfv_filtered["segment"] == "VIP"])
            vip_pct = vip_count / total_seg * 100
            vip_avg_monetary = rfv_filtered[rfv_filtered["segment"] == "VIP"]["monetary"].mean()
            st.success(
                f":bulb: **Insight:** Clientes **VIP** representam apenas **{vip_pct:.1f}%** "
                f"da base ({vip_count:,}), mas gastam em média **R$ {vip_avg_monetary:,.0f}** cada. "
                f"São o segmento mais valioso para o negócio."
            )

        risk_count = len(rfv_filtered[rfv_filtered["segment"] == "Em Risco"])
        risk_pct = risk_count / total_seg * 100
        if risk_pct > 0:
            st.warning(
                f":warning: **Alerta:** **{risk_pct:.1f}%** dos clientes ({risk_count:,}) "
                f"estão no segmento **Em Risco** — não compram há muito tempo. "
                f"Uma campanha de reativação pode recuperar parte desses clientes."
            )

        st.markdown("---")
        st.subheader("Boxplots RFV por Segmento")

        c1, c2, c3 = st.columns(3)
        for col, name in zip(["recency", "frequency", "monetary"], ["Recência (dias)", "Frequência (pedidos)", "Valor (R$)"]):
            with [c1, c2, c3][["recency", "frequency", "monetary"].index(col)]:
                fig_box = px.box(
                    rfv_filtered,
                    x="segment",
                    y=col,
                    color="segment",
                    title=name,
                    color_discrete_map=cluster_colors,
                    points=False,
                )
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.subheader("Gasto Total vs Recência (Bolhas = Frequência)")
        fig_scatter = px.scatter(
            rfv_filtered,
            x="recency",
            y="monetary",
            size="frequency",
            color="segment",
            hover_data=["customer_id"],
            title="Segmentação RFV",
            color_discrete_map=cluster_colors,
            labels={
                "recency": "Recência (dias desde última compra)",
                "monetary": "Gasto Total (R$)",
                "frequency": "Frequência (pedidos)",
            },
        )
        fig_scatter.update_layout(
            xaxis_title="Recência (dias)",
            yaxis_title="Gasto Total (R$)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Top 20 Clientes por RFV Score")
        top_rfm = rfv_filtered.sort_values("rfm_score", ascending=False).head(20)
        top_rfm_display = top_rfm[
            ["customer_id", "segment", "recency", "frequency", "monetary", "rfm_score"]
        ].reset_index(drop=True)
        top_rfm_display.index = top_rfm_display.index + 1
        st.dataframe(top_rfm_display, use_container_width=True)

        if full_mode:
            st.markdown("---")
            st.subheader("Análise Detalhada de Freq. vs Valor")
            bins_freq = [0, 1, 2, 3, 5, 10, 50, 200]
            labels_freq = ["1", "2", "3-4", "5-9", "10-19", "20+"]
            rfv_filtered["freq_group"] = pd.cut(
                rfv_filtered["frequency"],
                bins=bins_freq,
                labels=labels_freq,
                right=False,
            )
            freq_val = (
                rfv_filtered.groupby("freq_group", observed=True)
                .agg(
                    clientes=("customer_id", "count"),
                    receita_media=("monetary", "mean"),
                )
                .reset_index()
            )
            fig_freq = make_subplots(specs=[[{"secondary_y": True}]])
            fig_freq.add_trace(
                go.Bar(
                    x=freq_val["freq_group"],
                    y=freq_val["clientes"],
                    name="Clientes",
                    opacity=0.4,
                ),
                secondary_y=False,
            )
            fig_freq.add_trace(
                go.Scatter(
                    x=freq_val["freq_group"],
                    y=freq_val["receita_media"],
                    mode="lines+markers",
                    name="Receita Média (R$)",
                    line=dict(color="red", width=3),
                    marker=dict(size=10),
                ),
                secondary_y=True,
            )
            fig_freq.update_layout(title="Clientes vs Receita Média por Faixa de Frequência")
            fig_freq.update_yaxes(title_text="Clientes", secondary_y=False)
            fig_freq.update_yaxes(title_text="Receita Média (R$)", secondary_y=True)
            st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.warning("Dados RFV não disponíveis.")

# =============================================================================
# TAB 4 - PRODUTOS
# =============================================================================
with tab4:
    if full_mode:
        cat_perf = category_performance(df_filtered)

        st.subheader("Top Categorias por Review")
        c1, c2 = st.columns(2)
        with c1:
            top_cat = cat_perf.head(15)
            fig_top = px.bar(
                top_cat,
                x="avg_review",
                y="product_category_name_english",
                color="total_orders",
                orientation="h",
                title="Melhores Reviews",
                color_continuous_scale="Greens",
                text_auto=".2f",
            )
            fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig_top, use_container_width=True)
        with c2:
            bottom_cat = cat_perf.tail(15)
            fig_bottom = px.bar(
                bottom_cat,
                x="avg_review",
                y="product_category_name_english",
                color="total_orders",
                orientation="h",
                title="Piores Reviews",
                color_continuous_scale="Reds_r",
                text_auto=".2f",
            )
            fig_bottom.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig_bottom, use_container_width=True)

        best_cat = cat_perf.iloc[0]["product_category_name_english"]
        worst_cat = cat_perf.iloc[-1]["product_category_name_english"]
        st.info(
            f":bulb: **Insight:** A categoria com melhor avaliação é **{best_cat}** "
            f"({cat_perf.iloc[0]['avg_review']:.2f}) e a pior é **{worst_cat}** "
            f"({cat_perf.iloc[-1]['avg_review']:.2f}). "
            f"Considere investigar a qualidade/produtos da categoria **{worst_cat}**."
        )

        st.markdown("---")
        st.subheader("Receita vs Review por Categoria")

        fig_cat = px.scatter(
            cat_perf,
            x="total_sales",
            y="avg_review",
            size="total_orders",
            color="avg_review",
            hover_name="product_category_name_english",
            title="Receita Total vs Review Score (tamanho = pedidos)",
            labels={
                "total_sales": "Receita Total (R$)",
                "avg_review": "Review Médio",
                "total_orders": "Pedidos",
            },
            color_continuous_scale="RdYlGn",
        )
        fig_cat.add_hline(
            y=cat_perf["avg_review"].mean(),
            line_dash="dash",
            annotation_text=f"Review médio: {cat_perf['avg_review'].mean():.2f}",
        )
        fig_cat.add_vline(
            x=cat_perf["total_sales"].median(),
            line_dash="dash",
            annotation_text=f"Receita mediana: R$ {cat_perf['total_sales'].median():,.0f}",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")
        st.subheader("Tabela Completa de Categorias")
        cat_display = cat_perf.reset_index(drop=True)
        cat_display.index = cat_display.index + 1

        search_cat = st.text_input("Buscar categoria", "")
        if search_cat:
            cat_display = cat_display[
                cat_display["product_category_name_english"]
                .str.contains(search_cat, case=False, na=False)
            ]

        st.dataframe(cat_display, use_container_width=True)
    else:
        st.info("Carregue os dados completos para ver análise de produtos.")

# =============================================================================
# TAB 5 - RECOMENDAÇÃO
# =============================================================================
with tab5:
    st.subheader(":dart: Sistema de Recomendação de Produtos")
    st.markdown(
        "Encontre produtos similares ou receba recomendações personalizadas "
        "baseadas no histórico do cliente."
    )

    if full_mode:
        mode = st.radio(
            "Modo",
            ["Recomendar para Cliente", "Produtos Similares"],
            horizontal=True,
        )

        if mode == "Recomendar para Cliente":
            sample_customers = df_filtered["customer_unique_id"].sample(1000).tolist()
            customer_input = st.selectbox(
                "Selecione um cliente",
                options=[""] + sorted(sample_customers),
                format_func=lambda x: "Selecione um cliente..." if x == "" else x,
            )

            if customer_input:
                customer_data = df_filtered[df_filtered["customer_unique_id"] == customer_input]
                orders_count = customer_data["order_id"].nunique()
                total_spent = customer_data["payment_value"].sum()
                cities = customer_data["customer_city"].unique()

                with st.expander(":bust_in_silhouette: Dados do Cliente", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pedidos Realizados", orders_count)
                    c2.metric("Gasto Total", f"R$ {total_spent:,.2f}")
                    c3.metric("Cidade", ", ".join(cities[:3]))

                recs = recommend_for_customer(df_filtered, customer_input)
                if recs:
                    st.success(f"**{len(recs)} recomendações encontradas!**")
                    rec_data = []
                    for pid, score in recs:
                        info = df_filtered[df_filtered["product_id"] == pid]
                        if not info.empty:
                            row = info.iloc[0]
                            rec_data.append({
                                "Product ID": pid,
                                "Categoria": row.get("product_category_name_english", ""),
                                "Review": row.get("review_score", ""),
                                "Preço": f"R$ {row.get('price', 0):,.2f}",
                                "Score": score,
                            })
                    st.dataframe(pd.DataFrame(rec_data), use_container_width=True)
                else:
                    st.warning("Nenhuma recomendação encontrada para este cliente.")

        else:
            sample_products = df_filtered["product_id"].dropna().unique()
            product_input = st.selectbox(
                "Selecione um produto",
                options=[""] + sorted(sample_products[:1000]),
                format_func=lambda x: "Selecione um produto..." if x == "" else x,
            )

            if product_input:
                prod_info = df_filtered[df_filtered["product_id"] == product_input]
                if not prod_info.empty:
                    row = prod_info.iloc[0]
                    with st.expander(":package: Dados do Produto", expanded=True):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Categoria", row.get("product_category_name_english", ""))
                        c2.metric("Review", f"{row.get('review_score', '')}/5")
                        c3.metric("Preço", f"R$ {row.get('price', 0):,.2f}")

                sim = recommend_by_category(df_filtered, product_input)
                if sim:
                    st.success(f"**{len(sim)} produtos similares encontrados!**")
                    sim_data = []
                    for pid in sim:
                        info = df_filtered[df_filtered["product_id"] == pid]
                        if not info.empty:
                            row = info.iloc[0]
                            sim_data.append({
                                "Product ID": pid,
                                "Categoria": row.get("product_category_name_english", ""),
                                "Review": row.get("review_score", ""),
                                "Preço": f"R$ {row.get('price', 0):,.2f}",
                            })
                    st.dataframe(pd.DataFrame(sim_data), use_container_width=True)
                else:
                    st.warning("Nenhum produto similar encontrado.")

        st.markdown("---")
        estado_recomendacao = st.selectbox(
            "Filtrar recomendações por estado",
            options=["Todos"] + sorted(estados),
        )
        st.caption(
            ":bulb: O sistema de recomendação usa similaridade entre categorias "
            "para sugerir produtos. Para recomendações mais precisas, "
            "implemente filtragem colaborativa com o pacote `Surprise`."
        )
    else:
        st.info("Carregue os dados completos para usar o sistema de recomendação.")

st.markdown("---")
st.caption(
    "Projeto de portfólio — Dados: Olist Brazilian E-commerce (Kaggle) | "
    f"Dashboard gerado em {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}"
)
