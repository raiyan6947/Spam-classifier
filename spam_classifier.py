# spam_classifier.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

#Loading Dataset

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])

#Vectorize Text

vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Save the vectorizer for future use
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


#Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#SpamClassifier Class

class SpamClassifier:
    def __init__(self, model=None):
        self.model = model
        self.encoder = LabelEncoder()

    def train(self, X, y):
        try:
            # Encode labels
            self.encoder.fit(y)
            y_encoded = self.encoder.transform(y)
            # Train model
            self.model.fit(X, y_encoded)
            # Save trained model
            joblib.dump(self.model, "spam_classifier_model.pkl")
            joblib.dump(self.encoder, "label_encoder.pkl")
            print("Training complete and model saved!")
        except Exception as e:
            print(f"Error during training: {e}")

    def predict(self, X):
        try:
            # Load model if not loaded
            if self.model is None:
                self.model = joblib.load("spam_classifier_model.pkl")
                self.encoder = joblib.load("label_encoder.pkl")
            predictions = self.model.predict(X)
            return self.encoder.inverse_transform(predictions)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return []
            

#Training the Model

classifier = SpamClassifier(MultinomialNB())
classifier.train(X_train, y_train)

#Evaluating Model

predictions = classifier.predict(X_test)
accuracy = (predictions == y_test).mean() * 100
print(f"Test Accuracy: {accuracy:.2f}%")

#Predicting New Messages

def check_message(msg):
    #Loading vectorizer
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    X_new = vectorizer.transform([msg])
    return classifier.predict(X_new)[0]

#Example
new_msg = "Claim your free lottery prize now!"
result = check_message(new_msg)
print(f"Message: '{new_msg}' → Prediction: {result}") 

