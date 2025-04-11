from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

def evaluate_predictions(true_labels, predicted_labels):
    """
    Calculate and print evaluation metrics.
    
    Args:
        true_labels (list): Ground truth sentiment labels.
        predicted_labels (list): Predicted sentiment labels.
    """
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, digits=3))
    
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return acc, f1

def load_predictions_from_csv(path):
    """
    Load a CSV file containing columns: id, text, true_label, predicted_label.
    
    Returns:
        y_true, y_pred (tuple): Lists of actual and predicted labels.
    """
    df = pd.read_csv(path)
    return df['label'], df['prediction_label']
