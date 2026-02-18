import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

class SpamClassifier:
    def __init__(self, model=None):
        self.model = model
        self.encoder = LabelEncoder()

    def train(self, X, y):
        try:
            # Validate input messages
            self.encoder.fit(y)
            y_encoded = self.encoder.transform(y)
            self.model.fit(X, y_encoded)
            # Save the model
            joblib.dump(self.model, 'spam_classifier_model.pkl')
        except Exception as e:
            print(f"Error during training: {e}")

    def predict(self, X):
        try:
            # Load the trained model if not already loaded
            if self.model is None:
                self.model = joblib.load('spam_classifier_model.pkl')
            predictions = self.model.predict(X)
            return self.encoder.inverse_transform(predictions)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return []  # Return an empty list on error

# Example usage:
# classifier = SpamClassifier(your_trained_model)
# classifier.train(training_data, training_labels)
# predictions = classifier.predict(test_data)
