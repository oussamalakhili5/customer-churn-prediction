"""
Data preprocessing functions for Customer Churn prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load customer churn dataset from CSV."""
    return pd.read_csv(filepath)


def clean_total_charges(df):
    """Convert TotalCharges to numeric and drop null values."""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(how='any', inplace=True)
    return df


def create_tenure_bins(df):
    """Create tenure_bin categorical feature."""
    df = df.copy()
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    df['tenure_bin'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    return df


def encode_features(df):
    """One-hot encode categorical variables."""
    df = df.copy()
    return pd.get_dummies(df, drop_first=True)


def scale_features(X_train, X_test):
    """Apply StandardScaler to features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler