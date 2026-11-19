# IngProjectXAI
Reliability of XAI methods in medial imaging (ING Project).

## Requirements
- Python 3.13.1

## Setup (Windows)
``` bash
git clone https://github.com/juraj-n/IngProjectXAI.git
cd IngProjectXAI
py -3.13.1 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
The **C-NMC 2019 Dataset** (Classification of Normal vs. Malignant Cells in B-ALL White Blood Cancer Microscopic Images), sourced from Kaggle << [C-NMC 2019](https://www.kaggle.com/datasets/shafiullahshafin/c-nmc-2019-dataset?resource=download) >>

### Downloading the dataset
```bash
python src/scripts/download_cnmc_dataset.py
```