import re

def tokenize(text):
    """Clean and tokenize text input."""
    if text:
        text = re.sub(r'http\S+|www\S+|[^A-Za-z\s]', '', text)
        return text.lower().split()
    return []

def decode_prediction(label):
    """Convert numeric label to sentiment string."""
    labels = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}
    return labels.get(int(label), "Unknown")
