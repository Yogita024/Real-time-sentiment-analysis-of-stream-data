from pymongo import MongoClient
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Map labels
label_map = {
    "Negative": 0,
    "Positive": 1,
    "Neutral": 2,
    "Irrelevant": 3
}

def fetch_predictions_from_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["sentiment_streaming"]
    collection = db["predictions"]
    data = list(collection.find({}, {"_id": 0}))  # omit _id for clean dataframe
    return pd.DataFrame(data)

def evaluate_model():
    df = fetch_predictions_from_mongo()

    if df.empty or "label" not in df or "prediction_label" not in df:
        print("❌ No valid data found in MongoDB.")
        return

    df = df.dropna(subset=["label", "prediction_label"])
    y_true = df["label"].map(label_map)
    y_pred = df["prediction_label"].map(label_map)

    print("\n📊 Evaluation Report:")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))
    print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    evaluate_model()
