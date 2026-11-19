import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)


import kagglehub

path = kagglehub.dataset_download("shafiullahshafin/c-nmc-2019-dataset")

print("Path to dataset files:", path)