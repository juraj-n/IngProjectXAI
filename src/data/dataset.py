from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CNMCDataset(Dataset):
    def __init__(self, csv_path, data_root, split, transform=None):
        df = pd.read_csv(csv_path, sep=";")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.data_root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["malignant"])