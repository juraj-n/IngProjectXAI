import re
import random
from pathlib import Path
import pandas as pd

root = next(Path("data").rglob("C-NMC_training_data"))
dataset_root = root.parent
pattern = re.compile(r"^(UID_H?\d+)_", re.IGNORECASE)

rows = []
for fold in sorted(root.glob("fold_*")):
    for label_name, label in (("hem", 0), ("all", 1)):
        for f in sorted((fold / label_name).glob("*.bmp")):
            pid = pattern.match(f.name).group(1).upper()
            rows.append({
                "path": f.relative_to(dataset_root).as_posix(),
                "malignant": label,
                "patient_id": pid,
                "fold": int(fold.name.split("_")[1]),
            })

df = pd.DataFrame(rows)

# Split patients
images = df.groupby("patient_id").size().to_dict()
label_of = df.groupby("patient_id")["malignant"].first().to_dict()
random.seed(81)
split_of = {}
targets = {"train": 0.70, "val": 0.15, "test": 0.15}

for label in (0, 1):
    patients = sorted(p for p in images if label_of[p] == label)
    random.shuffle(patients)
    total = sum(images[p] for p in patients)
    got = {s: 0 for s in targets}
    for p in patients:
        s = max(targets, key=lambda k: targets[k] * total - got[k])
        split_of[p] = s
        got[s] += images[p]

df["split"] = df["patient_id"].map(split_of)

# Check: no patient in more than one split
assert (df.groupby("patient_id")["split"].nunique() == 1).all(), "Patient overlap!"
print("OK: no patient overlap between splits")
print(pd.crosstab(df["split"], df["malignant"], margins=True))

df.to_csv("data/cnmc_2019_metadata.csv", sep=";", index=False)
print("Saved", len(df), "rows")