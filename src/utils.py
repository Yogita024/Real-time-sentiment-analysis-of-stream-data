from pyspark.sql import SparkSession
from pymongo import MongoClient

def create_spark_session(app_name="SentimentApp"):
    """
    Initialize and return a SparkSession with Kafka support.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
        .getOrCreate()

def get_mongo_collection(uri="mongodb://localhost:27017", db_name="sentiment_db", collection_name="predictions"):
    """
    Connect to MongoDB and return the specified collection.
    """
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]
