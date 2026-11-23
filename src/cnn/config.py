from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "datasets" / "shafiullahshafin" / "c-nmc-2019-dataset" / "versions" / "1" / "C-NMC 2019 (PKG)"
METADATA_CSV = PROJECT_ROOT / "data" / "cnmc_2019_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"

# Training settings
SEED = 81
BATCH_SIZE = 32             # GPU out of memory -> 16
LEARNING_RATE = 1e-4
EPOCHS = 1                  # 10
NUM_WORKERS = 2             # Windows errors -> 0
IMAGE_SIZE = 224