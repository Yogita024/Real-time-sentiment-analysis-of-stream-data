# Real_Time_Sentiment_Analysis-Data-Processing

This repository contains the data processing components of a real-time sentiment analysis application for Twitter data. The application utilizes Apache Kafka, PySpark, and MongoDB.


![Screenshot 2024-05-24 193722](https://github.com/elmezianech/Real_Time_Sentiment_Analysis-Data-Processing/assets/120784838/627ee086-c82b-4752-a5c8-f6411b29e612)

## Introduction

This project aims to develop a real-time sentiment analysis application for tweets using Apache Kafka and Spark Streaming. The goal is to predict the sentiment (positive, negative, neutral, or irrelevant) of a given tweet.

## Architecture and Technologies Used

### Architecture
![image](https://github.com/user-attachments/assets/6c3efe88-01fb-46ef-bc3e-4d4ba332335b)

The architecture of the project consists of the following elements:
1. **Tweet Data Ingestion**: The pipeline begins by reading tweets from a CSV file named `twitter_validation.csv`.
2. **Apache Kafka**: Kafka functions as the message broker, handling the real-time flow of tweets for downstream processing.
3. **Apache Spark Streaming**: Spark Streaming consumes data from Kafka and performs the following tasks:
   - **Data Preprocessing**: Cleans and prepares the tweet text, extracting important features.
   - **Model Training**: A supervised machine learning model (Logistic Regression) is trained using labeled data from `twitter_training.csv`.
   - **Sentiment Prediction**: Applies the trained model to classify the sentiment of incoming tweets.
   - **Data Storage**: Stores the sentiment predictions into a MongoDB database.
4. **Web Application Integration**: The stored results are accessed by the `real-time-sentiment-analysis-web` repository for visual representation and analysis.


### Tools and Technologies

The tools and technologies used in this project include:
### Tools and Technologies
This project leverages a range of tools and technologies across various components to enable real-time sentiment analysis:
- **Python**: Serves as the primary programming language for building data processing workflows, training machine learning models, and integrating with external systems.
- **Docker**: Used to containerize individual services and components, ensuring consistent development environments and simplified deployment across different platforms.
- **Apache Kafka**: Acts as the real-time data streaming platform, facilitating the continuous flow of tweet data into the processing pipeline.
- **Apache Spark (PySpark)**: Handles large-scale data processing tasks and is responsible for training the machine learning model using distributed computing.
- **MongoDB**: A NoSQL database used for storing the sentiment analysis results generated by the model, enabling fast and flexible querying.
- **NLTK (Natural Language Toolkit)**: Employed for text preprocessing tasks such as tokenization, removing stop words, and lemmatizing the tweet content before feeding it into the model.
- **Matplotlib**: Utilized for creating visual representations of the data and analysis results, aiding in evaluation and interpretation.
- **Frontend and Backend Integration**: The project includes the `Real_Time_Sentiment_Analysis-Frontend-and-Backend` repository, which provides a complete solution for visualizing and interacting with the sentiment predictions via a web interface.

  
## Implementation

### Spark and Model Training

1. **Loading Training Data**: The dataset from `twitter_training.csv` is imported using PySpark for processing.
2. **Preprocessing the Data**: The data undergoes cleaning steps such as tokenization, removal of stop words, and lemmatization using the NLTK library.
3. **Training the Model**: A supervised learning algorithm—Logistic Regression—is applied to the preprocessed dataset for model training.
4. **Evaluating and Saving the Model**: After assessment, the most effective model is selected and saved for use in real-time sentiment prediction.


### Kafka

1. **Broker, Topic, and Partition Setup**: Kafka is configured with the necessary brokers, topics, and partitions for processing Twitter data.
2. **Kafka Streams**: Kafka Streams are used to read Twitter data from the `twitter_validation.csv` file.
3. **Real-Time Processing**: Incoming data is processed using the pre-trained machine learning model to predict sentiments.
4. **Result Storage**: Sentiment prediction results are saved in MongoDB.

### Integration with Web Application

1. **Frontend and Backend Integration**: The `real-time-sentiment-analysis-web` repository is included within this repository.
2. **Data Flow**: The data processed and predicted in this repository is used by the web application for visualization and user interaction.
