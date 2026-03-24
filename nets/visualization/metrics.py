import numpy as np


def compute_confusion_matrix(preds, targets, num_classes):

    # ---- ensure correct format
    preds = np.argmax(preds, axis=1)

    targets = targets.reshape(-1)
    targets = targets.astype(np.int32)
    preds = preds.astype(np.int32)

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for i in range(len(targets)):
        t = targets[i]
        p = preds[i]

        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    return cm