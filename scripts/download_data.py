"""
Download Olist dataset from Kaggle or local source.

Usage:
    python scripts/download_data.py --kaggle  # Download from Kaggle
    python scripts/download_data.py --local path/to/csvs  # Copy from local dir
"""

import argparse
import shutil
from pathlib import Path

DATA_RAW = Path("data/raw")


def download_kaggle() -> None:
    try:
        import kaggle
    except ImportError:
        print("Instale kaggle: pip install kaggle")
        print("Configure sua API key em ~/.kaggle/kaggle.json")
        return

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(
        "olistbr/brazilian-ecommerce",
        path=str(DATA_RAW),
        unzip=True,
    )
    print(f"Dados baixados em {DATA_RAW}")


def copy_local(source: str) -> None:
    src = Path(source)
    if not src.exists():
        print(f"Diretório {source} não encontrado")
        return

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, DATA_RAW / csv_file.name)
        print(f"Copiado: {csv_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true", help="Download do Kaggle")
    parser.add_argument("--local", type=str, help="Copiar de diretório local")
    args = parser.parse_args()

    if args.kaggle:
        download_kaggle()
    elif args.local:
        copy_local(args.local)
    else:
        print("Use --kaggle ou --local <path>")
