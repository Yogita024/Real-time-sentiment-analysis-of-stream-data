from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
from pymongo import MongoClient

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

def load_predictions_from_mongo():
    """
    Fetch prediction data from MongoDB and return true and predicted labels.
    
    Returns:
        tuple: Lists of actual and predicted labels.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["sentiment_streaming"]
    collection = db["predictions"]

    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)

    if df.empty or "label" not in df or "prediction_label" not in df:
        print("❌ No valid data found in MongoDB.")
        return [], []

    return df["label"].tolist(), df["prediction_label"].tolist()

if __name__ == "__main__":
    y_true, y_pred = load_predictions_from_mongo()
    if y_true and y_pred:
        evaluate_predictions(y_true, y_pred)
