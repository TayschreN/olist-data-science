import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

REPORTS_DIR = Path("reports/figures")


def plot_rfv_distribution(rfv: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i, col in enumerate(["recency", "frequency", "monetary"]):
        sns.histplot(rfv[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribuição - {col.capitalize()}")

    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / "rfv_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_clusters(rfv_clusters: pd.DataFrame, pca_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = sns.color_palette("Set2", n_colors=rfv_clusters["cluster"].nunique())
    sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue=rfv_clusters["cluster"],
        palette=colors,
        ax=axes[0],
        alpha=0.6,
    )
    axes[0].set_title("Clusters (PCA)")

    cluster_stats = rfv_clusters.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    cluster_stats.plot(kind="bar", ax=axes[1], colormap="Set2")
    axes[1].set_title("Perfil médio por cluster")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORTS_DIR / "clusters.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_monthly_trend(trend: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["order_month"],
            y=trend["total_revenue"],
            mode="lines+markers",
            name="Receita",
            yaxis="y",
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
        title="Receita e Pedidos por Mês",
        xaxis_title="Mês",
        yaxis_title="Receita (R$)",
        yaxis2=dict(overlaying="y", side="right"),
        hovermode="x unified",
    )
    fig.write_html(REPORTS_DIR / "monthly_trend.html")
    fig.show()


def plot_geo_map(df: pd.DataFrame) -> None:
    state_orders = (
        df.drop_duplicates(subset=["order_id", "customer_state"])
        .groupby("customer_state")
        .size()
        .reset_index(name="total_orders")
    )

    fig = px.choropleth(
        state_orders,
        locations="customer_state",
        locationmode="Brazil",
        color="total_orders",
        title="Pedidos por Estado",
        color_continuous_scale="Blues",
    )
    fig.write_html(REPORTS_DIR / "geo_map.html")
    fig.show()
