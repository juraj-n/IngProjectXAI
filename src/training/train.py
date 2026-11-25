import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import CNMCDataset
from src.training.evaluate import evaluate_model
from src.metrics.metrics import compute_metrics
from src.models.resnet import build_model
from src.data.transforms import get_eval_transforms, get_train_transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_main():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    train_ds = CNMCDataset(config.METADATA_CSV, config.DATA_ROOT, "train", get_train_transforms())
    val_ds = CNMCDataset(config.METADATA_CSV, config.DATA_ROOT, "val", get_eval_transforms())
    print(f"Images: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)

    # --- Class imbalance: rarer class gets a larger weight in the loss ---
    counts = train_ds.df["malignant"].value_counts().sort_index().to_numpy()  # [HEM, ALL]
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)
    print("Class counts [HEM, ALL]:", counts.tolist(), "-> loss weights:", weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # --- Model + optimizer + scheduler ---
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    # Learning rate shrinks smoothly from LEARNING_RATE to ~0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_f1 = -1.0
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        # ----- training -----
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()                # clear previous gradients
            outputs = model(images)              # model's predictions
            loss = criterion(outputs, labels)    # how wrong is the model?
            loss.backward()                      # work out how to change the weights
            optimizer.step()                     # change the weights

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)
        train_loss = running_loss / seen
        train_acc = correct / seen

        # ----- validation -----
        val_loss, val_labels, val_preds = evaluate_model(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_preds)

        current_lr = optimizer.param_groups[0]["lr"]  # LR used during this epoch
        scheduler.step()                              # lower the LR for the next epoch

        print(f"Epoch {epoch:02d}/{config.EPOCHS} | lr {current_lr:.2e} | "
              f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_metrics['accuracy']:.4f} | "
              f"val F1 {val_metrics['f1_macro']:.4f}")

        history.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_metrics["accuracy"],
            "Validation F1": val_metrics["f1_macro"],
        })
        pd.DataFrame(history).to_csv(config.OUTPUT_DIR / "history.csv", sep=";", decimal=",", index=False)

        # ----- save best checkpoint -----  
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), config.OUTPUT_DIR / "best.pt")
            print("  -> new best model saved")

    print(f"\nDone. Best validation F1: {best_f1:.4f}")
    print("Now run: python -m src.training.evaluate")


if __name__ == "__main__":
    train_main()