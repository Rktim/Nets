import numpy as np


# -------------------------
# BASIC
# -------------------------
def accuracy(logits, y_true):
    preds = np.argmax(logits.data, axis=1)
    return (preds == y_true.data).mean()


def precision(logits, y_true):
    cm = confusion_matrix(logits, y_true, num_classes=logits.data.shape[1])

    precisions = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precisions.append(tp / (tp + fp + 1e-9))

    return np.mean(precisions)


def recall(logits, y_true):
    cm = confusion_matrix(logits, y_true, num_classes=logits.data.shape[1])

    recalls = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recalls.append(tp / (tp + fn + 1e-9))

    return np.mean(recalls)


def f1_score(logits, y_true):
    p = precision(logits, y_true)
    r = recall(logits, y_true)
    return 2 * p * r / (p + r + 1e-9)

# -------------------------
# CONFUSION MATRIX
# -------------------------
def confusion_matrix(logits, y_true, num_classes):

    preds = np.argmax(logits.data, axis=1)
    y = y_true.data

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y, preds):
        cm[int(t), int(p)] += 1

    return cm


# -------------------------
# CLASSIFICATION REPORT
# -------------------------
def classification_report(logits, y_true, num_classes):

    cm = confusion_matrix(logits, y_true, num_classes)

    report = {}

    for i in range(num_classes):

        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        report[i] = {
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    return report


# -------------------------
# ROC-AUC (binary only)
# -------------------------
def roc_auc(logits, y_true):

    probs = logits.data
    if probs.shape[1] > 1:
        probs = probs[:, 1]  # positive class

    y = y_true.data

    # sort by probability
    sorted_idx = np.argsort(probs)
    y = y[sorted_idx]

    tpr = []
    fpr = []

    P = np.sum(y == 1)
    N = np.sum(y == 0)

    tp = 0
    fp = 0

    for i in range(len(y)-1, -1, -1):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr.append(tp / (P + 1e-9))
        fpr.append(fp / (N + 1e-9))

    # trapezoidal rule
    auc = 0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return abs(auc)