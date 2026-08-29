from setuptools import setup, find_packages

setup(
    name="customer-churn-prediction",
    version="0.1.0",
    description="End-to-end machine learning project for customer churn prediction",
    author="Oussama Lakhili",
    author_email="oussama.lakhili@outlook.fr",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "imbalanced-learn>=0.8.0",
        "streamlit>=1.10.0",
        "joblib>=1.1.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)