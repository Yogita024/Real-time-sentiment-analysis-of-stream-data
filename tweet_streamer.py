import pandas as pd
import json
import logging
import time
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO)

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

TOPIC = 'twitter'

def handle_success(metadata):
    logging.info(f"Delivered to {metadata.topic} [partition {metadata.partition}] at offset {metadata.offset}")

def handle_failure(exception):
    logging.error("Delivery failed", exc_info=exception)

def stream_data_from_csv(path):
    columns = ["Tweet ID", "Entity", "Sentiment", "TweetContent"]
    df = pd.read_csv(path, names=columns)

    for _, row in df.iterrows():
        message = {
            "id": row["Tweet ID"],
            "entity": row["Entity"],
            "text": row["TweetContent"].replace(',', ' '),
            "label": row["Sentiment"]
        }
        producer.send(TOPIC, value=message).add_callback(handle_success).add_errback(handle_failure)
        time.sleep(1)

if __name__ == "__main__":
    stream_data_from_csv("../data/twitter_validation.csv")
