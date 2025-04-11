from evaluate import evaluate_predictions, load_predictions_from_csv

# Load predictions
y_true, y_pred = load_predictions_from_csv("data/predictions/part-*.csv")

# Print evaluation report
evaluate_predictions(y_true, y_pred)
