# Spam Message Classifier (Naive Bayes + TF-IDF)

## Overview
This project implements a machine learning model that classifies SMS messages as **Spam** or **Ham (Not Spam)** using TF-IDF vectorization and the Multinomial Naive Bayes algorithm.

The model is trained on a real-world SMS dataset containing more than 5,000 labeled messages.

---

## Problem Statement
Spam messages are a major issue in digital communication.  
The objective of this project is to build a machine learning model that can automatically detect whether a message is spam or legitimate.

---

## Dataset
- SMS Spam Collection Dataset  
- Approximately 5,500 labeled messages  
- Two categories: `spam` and `ham`

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

## Model Workflow
1. Load dataset  
2. Convert text into numerical features using TF-IDF  
3. Split dataset into training and testing sets  
4. Train Multinomial Naive Bayes classifier  
5. Evaluate model performance  
6. Predict new messages  

---

## Model Performance
Accuracy: 95–98% (may vary depending on train-test split)

Example:

Input: "Claim your free lottery prize now!"  
Prediction: Spam  

---

## How to Run

Install required libraries:
pip install pandas scikit-learn

### Run the Script
python spam_classifier.py

---

## Project Structure

Spam-Classifier/ │ ├── spam_classifier.py ├── README.md

---

## Future Improvements

- Compare with Logistic Regression
- Add advanced preprocessing (stemming, lemmatization)
- Save and load trained model using joblib
- Deploy as a web application using Flask
- Add a simple user interface

---

## Author

Developed as a machine learning mini-project for text classification practice.
