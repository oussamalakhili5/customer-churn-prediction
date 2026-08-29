# Customer Churn Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Overview

End-to-end machine learning project to predict customer churn in the telecommunications industry. The project follows the complete data science lifecycle: from business understanding through data cleaning, exploratory analysis, model building, hyperparameter optimization, and deployment via an interactive web application.

**Business Impact**: Identifying at-risk customers before they churn allows the company to implement targeted retention strategies, potentially saving millions in revenue.

## 🎯 Project Objectives

- Identify key factors driving customer churn
- Build a predictive model with high recall for the churn class
- Deploy an interactive web application for real-time predictions
- Handle class imbalance using advanced resampling techniques
- Optimize model performance using multiple hyperparameter tuning methods

## 📊 Dataset

The dataset contains information about 7,043 telecom customers with 21 features:

| Category | Features |
|----------|----------|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Account Info** | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| **Target** | Churn (Yes/No) |

**Churn Rate**: 26.53% (imbalanced dataset)

## 🏗️ Project Structurecustomer-churn-prediction/
│
├── data/
│ ├── raw/
│ │ └── Customer-Churn.csv # Raw dataset
│ └── processed/ # Cleaned data (if generated)
│
├── notebooks/
│ └── ML_Model_Building.ipynb # Full EDA & Model Development
│
├── src/
│ ├── init.py
│ ├── preprocessing.py # Data cleaning functions
│ ├── model.py # Model loading & prediction
│ └── utils.py # Utility functions
│
├── models/
│ ├── ada_boost_churn_model.pkl # AdaBoost with sample weighting
│ ├── best_xgboost_churn_model.pkl # XGBoost (RandomizedSearchCV)
│ └── best_optuna_churn_model.pkl # XGBoost (Optuna optimized)
│
├── app/
│ └── streamlit_app.py # Interactive web application
│
├── deployment/
│ └── Deployment.txt # Deployment instructions
│
├── tests/
│ └── test_model.py # Unit tests
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md


## 🔄 Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed distribution of all features
- Identified churn patterns across customer segments
- Visualized relationships between features and churn

### 2. Data Cleaning
- Converted `TotalCharges` from object to float64
- Identified and dropped 11 rows with null values
- Result: 7,032 clean records

### 3. Feature Engineering
- Created `tenure_bin` categorical feature (12-month intervals)
- Applied one-hot encoding to categorical variables
- Standardized numeric features using StandardScaler

### 4. Handling Class Imbalance

| Technique | Description |
|-----------|-------------|
| SMOTE | Synthetic Minority Over-sampling |
| SMOTEENN | SMOTE + Edited Nearest Neighbors |
| ADASYN | Adaptive Synthetic Sampling |
| Sample Weighting | Class weights in model training |
| scale_pos_weight | XGBoost built-in parameter |

### 5. Model Development

| Model | Best F1 Score | Notes |
|-------|--------------|-------|
| Decision Tree (Baseline) | 0.50 | Imbalanced data issue |
| Decision Tree + SMOTEENN | 0.60 | Improved recall |
| Random Forest + SMOTEENN | 0.62 | Better generalization |
| XGBoost + SMOTE | 0.64 | Strong performance |
| AdaBoost + Sample Weighting | 0.63 | Good balance |
| **XGBoost + Optuna** | **0.6449** | **Best model** |

### 6. Hyperparameter Optimization

#### RandomizedSearchCV
- 50 iterations × 5-fold cross-validation
- Scoring: F1 (for imbalanced data)
- Best parameters: `max_depth=7, learning_rate=0.01, n_estimators=400`

#### Optuna
- 100 trials × 5-fold cross-validation
- Bayesian optimization with Tree-structured Parzen Estimator
- Best F1: 0.6449

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/VOTRE_USERNAME/customer-churn-prediction.git

# Navigate to project
cd customer-churn-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

🖥️ Running the Web Application
streamlit run app/streamlit_app.py
Open http://localhost:8501 in your browser.
Features:
Interactive form to input customer details

Real-time churn prediction with probability

Preprocessing pipeline identical to training

Debug mode for transparency

📈 Key Insights
Contract Type: Month-to-month customers are more likely to churn

Tenure: Newer customers (1-12 months) have higher churn rate

Monthly Charges: Higher charges correlate with increased churn

Internet Service: Fiber optic users show higher churn tendency

Payment Method: Electronic check users are at higher risk

🛠️ Tech Stack
Python 3.8+

Pandas & NumPy: Data manipulation

Matplotlib & Seaborn: Visualization

Scikit-learn: ML algorithms & preprocessing

XGBoost: Gradient boosting

CatBoost & LightGBM: Alternative boosting frameworks

Imbalanced-learn: Resampling techniques

Optuna: Hyperparameter optimization

Streamlit: Web application

Joblib: Model serialization

📚 Skills Demonstrated
Data cleaning and preprocessing

Exploratory Data Analysis

Feature engineering

Handling imbalanced datasets

Multiple ML algorithms comparison

Hyperparameter tuning (RandomizedSearchCV & Optuna)

Model evaluation & selection

Building interactive web applications

Git version control

Professional project structure

📄 License
This project is licensed under the MIT License.

👤 Author
Votre Nom

GitHub: oussamalakhili5

LinkedIn: https://www.linkedin.com/in/oussama-lakhili-06aaa0234/



