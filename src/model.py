"""
Model loading, training, and prediction functions.
"""
import joblib
import numpy as np


def save_model(model, filepath):
    """Save trained model to .pkl file."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from .pkl file."""
    return joblib.load(filepath)


def predict_churn(model, X):
    """Predict churn probability and class."""
    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return pred, proba