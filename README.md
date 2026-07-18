# Olist Data Science

Análise RFV (Recência, Frequência e Valor), segmentação de clientes e sistema de recomendação utilizando o dataset público da [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Visão Geral

Projeto end-to-end de Ciência de Dados com dados reais de e-commerce brasileiro. O dataset contém ~100 mil pedidos de 2016 a 2018 em diversos estados do Brasil.

### Etapas

1. **EDA** — Análise exploratória com insights de negócio
2. **RFV Analysis** — Segmentação de clientes por Recência, Frequência e Valor
3. **Clusterização** — K-Means + PCA para identificar perfis de clientes
4. **Sistema de Recomendação** — Filtragem colaborativa e content-based
5. **Pipeline MLOps** — MLflow para tracking de experimentos
6. **Deploy** — Dashboard Streamlit + API FastAPI

## Tech Stack

| Ferramenta | Uso |
|---|---|
| Python / pandas / polars | Manipulação de dados |
| DuckDB | Consultas SQL analíticas |
| scikit-learn | Clusterização (K-Means, PCA) |
| Surprise / Implicit | Sistema de recomendação |
| Plotly / Seaborn | Visualizações |
| Streamlit | Dashboard interativo |
| FastAPI | API do modelo |
| MLflow | Experiment tracking |
| Docker | Containerização |
| pytest | Testes |

## Estrutura

```
olist-data-science/
├── data/              # Dados brutos e processados
├── notebooks/         # Análises exploratórias
├── src/               # Código modular
│   ├── etl.py         # Extração e limpeza
│   ├── features.py    # Feature engineering (RFV)
│   ├── models.py      # Modelos (K-Means)
│   └── recommender.py # Sistema de recomendação
├── app/               # Streamlit + FastAPI
├── tests/             # Testes unitários
├── docker/            # Dockerfile e compose
├── models/            # Modelos treinados
├── reports/           # Figuras e relatórios
└── scripts/           # Scripts utilitários
```

## Como usar

```bash
# Instalar dependências
pip install -e .

# Com optional (recomendação)
pip install -e ".[recommender]"

# Download dos dados
python scripts/download_data.py

# Executar pipeline completo
python scripts/run_pipeline.py

# Dashboard
streamlit run app/streamlit_app.py

# API
uvicorn app.api:app --reload
```

## Resultados

[Adicionar screenshots e resultados após execução]

## Licença

MIT
