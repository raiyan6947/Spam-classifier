import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Loading dataset directly from internet
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])

print("Dataset loaded successfully!")
print("Total samples:", len(df))

# 2. Converting text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Training model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluating model
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))


# 6. Function to check new messages
def check_message(msg):
    vec_msg = vectorizer.transform([msg])
    prediction = model.predict(vec_msg)
    return prediction[0]


# 7. Testing new message
new_message = "Claim your free lottery prize now before it expires!"
print("\nNew Message:", new_message)
print("Prediction:", check_message(new_message))