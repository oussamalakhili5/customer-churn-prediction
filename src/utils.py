"""
Utility functions for data analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def missing_values_table(df):
    """Return DataFrame with missing values percentage."""
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    return pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing,
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Percent', ascending=False)


def plot_churn_distribution(df):
    """Plot churn distribution."""
    plt.figure(figsize=(8, 6))
    df['Churn'].value_counts().plot(kind='barh', color=['green', 'red'])
    plt.xlabel("Count")
    plt.ylabel("Churn")
    plt.title("Churn Distribution")
    plt.show()