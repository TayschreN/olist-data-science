FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY data/ data/

RUN pip install --no-cache-dir -e .

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501"]
