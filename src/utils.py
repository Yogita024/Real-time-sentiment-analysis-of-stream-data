from pyspark.sql import SparkSession

def create_spark_session(app_name="SentimentApp"):
    """
    Initialize and return a SparkSession with Kafka support.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
        .getOrCreate()
