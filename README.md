SMS Spam Classifier (Machine Learning Project)

This is a simple machine learning project that classifies SMS messages as spam or ham (not spam) using Natural Language Processing (NLP) techniques.

The goal of this project was to understand how text data can be converted into numerical features and used to train a real classification model.

## Project Overview

Spam messages are a common problem in communication systems. In this project, I built a basic spam detection model using:

TF-IDF Vectorization for text feature extraction 

Multinomial Naive Bayes for classification

Scikit-learn for model training and evaluation

Joblib for saving and loading trained models
The model is trained on a publicly available SMS dataset and achieves high accuracy on unseen test data.

## How It Works:

Dataset Loading

The SMS dataset is loaded from an online source and contains labeled messages (spam or ham).

Text Preprocessing & Vectorization

Text is converted to lowercase
English stopwords are removed

Messages are transformed into numerical vectors using TF-IDF 

Model Training:

The dataset is split into training and testing sets

A Multinomial Naive Bayes classifier is trained

Labels are encoded using LabelEncoder
Model Saving

The trained model, label encoder, and vectorizer are saved as .pkl files for reuse.
Prediction

You can input a new message, and the model will predict whether it is spam or not.

## Example

Input:

"Claim your free lottery prize now!"

Output:

spam

## Technologies Used:

Python
Pandas
Scikit-learn
Joblib


## Files Generated:

After training, the following files are saved:

spam_classifier_model.pkl → Trained model

label_encoder.pkl → Encoded labels

tfidf_vectorizer.pkl → TF-IDF vectorizer

These files allow the model to be reused without retraining.

## What I Learned:

How text data is converted into numerical features

How Naive Bayes works for text classification

How to save and reuse trained ML models

Structuring a simple ML project in Python

## Future Improvements:

Add a simple web interface (Flask or Streamlit)

Improve preprocessing (stemming, lemmatization)

Try different models (Logistic Regression, SVM)

Deploy the model online

## This project is part of my journey into machine learning and practical NLP applications.