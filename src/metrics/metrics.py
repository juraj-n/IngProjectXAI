from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        # rows = true class (HEM, ALL), columns = predicted class (HEM, ALL)
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }