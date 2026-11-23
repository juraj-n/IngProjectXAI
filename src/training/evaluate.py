import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import CNMCDataset
from src.metrics.metrics import compute_metrics
from src.models.resnet import build_model
from src.data.transforms import get_eval_transforms


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    """Runs the model on a dataloader. Returns (loss, true labels, predictions)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / len(all_labels), all_labels, all_preds


def eval_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = CNMCDataset(config.METADATA_CSV, config.DATA_ROOT, "test", get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    model = build_model().to(device)
    checkpoint = config.OUTPUT_DIR / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    _, labels, preds = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
    results = compute_metrics(labels, preds)

    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test macro F1: {results['f1_macro']:.4f}")
    print("Confusion matrix (rows = true HEM/ALL, columns = predicted HEM/ALL):")
    print(results["confusion_matrix"])

    with open(config.OUTPUT_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    eval_main()