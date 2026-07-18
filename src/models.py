import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from pathlib import Path

MODELS_DIR = Path("models")


def segment_customers(
    rfv: pd.DataFrame, n_clusters: int = 4, save: bool = True
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    scaler = StandardScaler()
    rfv_scaled = scaler.fit_transform(rfv[["recency", "frequency", "monetary"]])

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    )
    clusters = kmeans.fit_predict(rfv_scaled)

    result = rfv.copy()
    result["cluster"] = clusters

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(kmeans, MODELS_DIR / "kmeans.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    return result, kmeans, scaler


def pca_transform(rfv: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    rfv_scaled = scaler.transform(rfv[["recency", "frequency", "monetary"]])
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(rfv_scaled)
    return pd.DataFrame(
        components, columns=["pc1", "pc2"], index=rfv.index
    )


def elbow_method(rfv: pd.DataFrame, max_k: int = 10) -> list[float]:
    scaler = StandardScaler()
    rfv_scaled = scaler.fit_transform(rfv[["recency", "frequency", "monetary"]])

    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfv_scaled)
        inertias.append(kmeans.inertia_)

    return inertias
