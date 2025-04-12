# Real_Time_Sentiment_Analysis-Data-Processing

This repository contains the data processing components of a real-time sentiment analysis application for Twitter data. The application utilizes Apache Kafka, PySpark, and MongoDB.

## Introduction

This project aims to develop a real-time sentiment analysis application for tweets using Apache Kafka and Spark Streaming. The goal is to predict the sentiment (positive, negative, neutral, or irrelevant) of a given tweet.

## Architecture and Technologies Used

### Architecture
![image](https://github.com/user-attachments/assets/8b4cc524-17c4-482b-80cd-b42d60739fa9)
<p align="center"><b> Tweet Analysis Architecture Diagram</b></p>


The architecture of the project consists of the following elements:
1. **Tweet Data Ingestion**: The pipeline begins by reading tweets from a CSV file named `twitter_sentiment_analysis_data.csv`.
2. **Apache Kafka**: Kafka functions as the message broker, handling the real-time flow of tweets for downstream processing.
3. **Apache Spark Streaming**: Spark Streaming consumes data from Kafka and performs the following tasks:
   - **Data Preprocessing**: Cleans and prepares the tweet text, extracting important features.
   - **Model Training**: A supervised machine learning model (Logistic Regression) is trained using labeled data.
   - **Sentiment Prediction**: Applies the trained model to classify the sentiment of incoming tweets.
   - **Data Storage**: Stores the sentiment predictions into a MongoDB database.
4. **Web Application Integration**: The stored results are accessed by the `real-time-sentiment-analysis-of-stream-data` repository for visual representation and analysis.


### Tools and Technologies

This project leverages a range of tools and technologies across various components to enable real-time sentiment analysis:
- **Python**: Serves as the primary programming language for building data processing workflows, training machine learning models, and integrating with external systems.
- **Docker**: Used to containerize individual services and components, ensuring consistent development environments and simplified deployment across different platforms.
- **Apache Kafka**: Acts as the real-time data streaming platform, facilitating the continuous flow of tweet data into the processing pipeline.
- **Apache Spark (PySpark)**: Handles large-scale data processing tasks and is responsible for training the machine learning model using distributed computing.
- **MongoDB**: A NoSQL database used for storing the sentiment analysis results generated by the model, enabling fast and flexible querying.
- **NLTK (Natural Language Toolkit)**: Employed for text preprocessing tasks such as tokenization, removing stop words, and lemmatizing the tweet content before feeding it into the model.
- **Matplotlib**: Utilized for creating visual representations of the data and analysis results, aiding in evaluation and interpretation.
  
## Implementation

### Spark and Model Training

1. **Loading Training Data**: The dataset from `twitter_sentiment_analysis_data.csv` is imported using PySpark for processing.
2. **Preprocessing the Data**: The data undergoes cleaning steps such as tokenization, removal of stop words, and lemmatization using the NLTK library.
3. **Training the Model**: A supervised learning algorithm—Logistic Regression—is applied to the preprocessed dataset for model training.
4. **Evaluating and Saving the Model**: After assessment, the most effective model is selected and saved for use in real-time sentiment prediction.


### Kafka

1. **Broker, Topic, and Partition Setup**: Kafka is configured with the necessary brokers, topics, and partitions for processing Twitter data.
2. **Kafka Streams**: Kafka Streams are used to read Twitter data from the `twitter_sentiment_analysis_data.csv` file.
3. **Real-Time Processing**: Incoming data is processed using the pre-trained machine learning model to predict sentiments.
4. **Result Storage**: Sentiment prediction results are saved in MongoDB.
