import re
from pyspark.sql.functions import from_json, udf, col
from pyspark.sql.types import StructType, StringType, ArrayType
from pyspark.ml import PipelineModel
from utils import create_spark_session, get_mongo_collection
from pyspark.sql.streaming import ForeachWriter

# Tokenization UDF
def tokenize(text):
    if text:
        text = re.sub(r"http\S+|www\S+|[^A-Za-z\s]", "", text)
        return text.lower().split()
    return []

# Label decoding UDF
def decode_prediction(label):
    mapping = {0: "Negative", 1: "Positive", 2: "Neutral", 3: "Irrelevant"}
    return mapping.get(int(label), "Unknown")

# MongoDB writer using ForeachWriter
class MongoSink(ForeachWriter):
    def open(self, partition_id, epoch_id):
        self.collection = get_mongo_collection()
        return True

    def process(self, row):
        self.collection.insert_one(row.asDict())

    def close(self, error):
        pass

if __name__ == "__main__":
    spark = create_spark_session("StreamingSentimentAnalysis")
    spark.sparkContext.setLogLevel("WARN")

    stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "twitter") \
        .option("startingOffsets", "latest") \
        .load() \
        .selectExpr("CAST(value AS STRING) as message")

    schema = StructType() \
        .add("id", StringType()) \
        .add("entity", StringType()) \
        .add("text", StringType()) \
        .add("label", StringType())

    data = stream.select(from_json(col("message"), schema).alias("json")).select("json.*")

    tokenize_udf = udf(tokenize, ArrayType(StringType()))
    decode_udf = udf(decode_prediction, StringType())

    cleaned = data.withColumn("tokens", tokenize_udf(col("text"))).dropna(subset=["tokens"])
    cleaned = cleaned.withColumn("cleaned_text", col("tokens").cast(StringType()))

    model = PipelineModel.load("../model/pre_trained_model")

    result = model.transform(cleaned.select("cleaned_text", "id", "entity", "label"))
    result = result.withColumn("prediction_label", decode_udf(col("prediction")))

    output = result.select("id", "entity", "cleaned_text", "label", "prediction_label")

    # Write to console
    query_console = output.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    # Write to MongoDB
    query_mongo = output.writeStream \
        .foreach(MongoSink()) \
        .outputMode("append") \
        .start()

    query_console.awaitTermination()
    query_mongo.awaitTermination()
