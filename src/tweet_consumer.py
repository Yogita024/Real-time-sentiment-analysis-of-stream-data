import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, udf, col
from pyspark.sql.types import StructType, StringType, ArrayType
from pyspark.ml import PipelineModel

def tokenize(text):
    if text:
        text = re.sub(r'http\S+|www\S+|[^A-Za-z\s]', '', text)
        return text.lower().split()
    return []

def decode_prediction(label):
    labels = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}
    return labels.get(int(label), "Unknown")

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("LiveSentimentAnalysis") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Kafka stream source
    stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "twitter") \
        .option("startingOffsets", "latest") \
        .load() \
        .selectExpr("CAST(value AS STRING) AS message")

    # Define input schema
    schema = StructType() \
        .add("id", StringType()) \
        .add("entity", StringType()) \
        .add("text", StringType()) \
        .add("label", StringType())

    # Extract structured fields from JSON string
    data = stream.select(from_json(col("message"), schema).alias("json")) \
                 .select("json.*")

    # Register UDFs
    tokenize_udf = udf(tokenize, ArrayType(StringType()))
    decode_udf = udf(decode_prediction, StringType())

    # Clean and transform
    cleaned = data.withColumn("tokens", tokenize_udf(col("text"))) \
                  .dropna(subset=["tokens"])

    cleaned = cleaned.withColumn("cleaned_text", col("tokens").cast(StringType()))

    # Load trained ML pipeline
    model = PipelineModel.load("../model/pre_trained_model")

    # Predict sentiment
    result = model.transform(cleaned.select("cleaned_text", "id", "entity", "label"))

    result = result.withColumn("prediction_label", decode_udf(col("prediction")))

    output = result.select("id", "entity", "cleaned_text", "label", "prediction_label")

    query = output.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    query.awaitTermination()
