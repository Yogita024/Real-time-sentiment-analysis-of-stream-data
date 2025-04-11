from evaluate import evaluate_predictions, load_predictions_from_mongo
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Load predictions
y_true, y_pred = load_predictions_from_mongo()

# Basic metrics
if y_true and y_pred:
    evaluate_predictions(y_true, y_pred)

    # Label binarization for ROC
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    y_pred_bin = lb.transform(y_pred)

    # Plot ROC curves
    plt.figure(figsize=(10, 7))
    for i, class_label in enumerate(lb.classes_):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
        plt.plot(fpr, tpr, label=f"{class_label} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No predictions found in MongoDB for evaluation.")
