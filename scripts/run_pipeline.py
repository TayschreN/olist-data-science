"""
Pipeline completo:
1. Carrega dados
2. ETL e merge
3. RFV + clusterização
4. Salva resultados
5. Plota gráficos
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.etl import load_raw_data, merge_datasets
from src.features import compute_rfv, rfm_score
from src.models import segment_customers, pca_transform, elbow_method
from src.analysis import (
    delivery_performance,
    category_performance,
    monthly_sales_trend,
)
from src.visualization import (
    plot_rfv_distribution,
    plot_clusters,
    plot_monthly_trend,
)


def run_pipeline() -> None:
    print("1. Carregando dados...")
    data = load_raw_data()
    if not data:
        print("ERRO: Nenhum dado encontrado em data/raw/")
        print("Use python scripts/download_data.py")
        return

    print(f"   Arquivos carregados: {list(data.keys())}")

    print("2. Merge dos datasets...")
    df = merge_datasets(data)
    print(f"   Shape: {df.shape}")

    print("3. Computando RFV...")
    rfv = compute_rfv(df)
    rfv = rfm_score(rfv)
    print(f"   Clientes únicos: {len(rfv)}")

    print("4. Método do cotovelo...")
    inertias = elbow_method(rfv)
    print(f"   Inércias para k=1..{len(inertias)}: {[round(i, 1) for i in inertias]}")

    print("5. Segmentando clientes...")
    rfv_clusters, model, scaler = segment_customers(rfv, n_clusters=4)
    cluster_counts = rfv_clusters["cluster"].value_counts().sort_index()
    print(f"   Clusters: {cluster_counts.to_dict()}")

    print("6. PCA para visualização...")
    pca_df = pca_transform(rfv, scaler)

    print("7. Análises de negócio...")
    delivery_perf = delivery_performance(df)
    cat_perf = category_performance(df)
    trend = monthly_sales_trend(df)
    print( cat_perf.head(10).to_string(index=False) )

    print("8. Gerando gráficos...")
    plot_rfv_distribution(rfv)
    plot_clusters(rfv_clusters, pca_df)
    plot_monthly_trend(trend)

    print("9. Salvando dados processados...")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / "olist_full.parquet")
    rfv_clusters.to_csv(processed_dir / "rfv_clusters.csv", index=False)
    delivery_perf.to_csv(processed_dir / "delivery_performance.csv", index=False)
    cat_perf.to_csv(processed_dir / "category_performance.csv", index=False)
    trend.to_csv(processed_dir / "monthly_trend.csv", index=False)

    print(" Pipeline concluído!")


if __name__ == "__main__":
    run_pipeline()
