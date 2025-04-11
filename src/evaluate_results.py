from evaluate import evaluate_predictions, load_predictions_from_mongo

# Load predictions from MongoDB
y_true, y_pred = load_predictions_from_mongo()

# Print evaluation report
if y_true and y_pred:
    evaluate_predictions(y_true, y_pred)
else:
    print("Could not evaluate — no predictions found.")
