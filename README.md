# 🛡️ Car Insurance Fraud Detection System

> **An End-to-End Machine Learning Project for Detecting Fraudulent Car Insurance Claims using Scikit-learn and Streamlit**

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Insurance fraud costs companies billions of dollars every year. Detecting fraudulent claims early helps insurance providers reduce financial losses and improve claim processing efficiency.

This project uses **Machine Learning** to predict whether a **car insurance claim** is **Fraudulent** or **Genuine**. The application includes a professional **Streamlit dashboard** that allows users to enter claim information and receive real-time predictions.

---

## 🚀 Features

- ✅ End-to-End Machine Learning Pipeline
- ✅ Data Cleaning & Preprocessing
- ✅ Feature Engineering
- ✅ Multiple Model Comparison
- ✅ Interactive Streamlit Dashboard
- ✅ Real-Time Fraud Prediction
- ✅ Analytics Dashboard
- ✅ Model Performance Comparison
- ✅ Professional Dark UI

---

## 📂 Dataset

**Dataset:** Car Insurance Fraud Claims Dataset

- Total Records: **1000**
- Total Features: **40**
- Target Variable: **fraud_reported**

### Target Distribution

| Class | Count |
|-------|------:|
| Genuine Claims | 753 |
| Fraudulent Claims | 247 |

The dataset is **imbalanced**, making **Recall** and **F1 Score** more important evaluation metrics than Accuracy.

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Git & GitHub

---

# 📊 Exploratory Data Analysis

Performed comprehensive EDA including:

- Missing Value Analysis
- Duplicate Detection
- Fraud Distribution
- Incident Severity Analysis
- Claim Amount Analysis
- Collision Type Analysis
- Property Damage Analysis
- Police Report Analysis
- Correlation Analysis

---

# ⚙️ Feature Engineering

Created meaningful business features including:

- Policy Duration
- Vehicle Age
- Claim per Vehicle Age
- Property Claim Ratio
- Vehicle Claim Ratio
- Injury Claim Ratio
- Claim per Vehicle

These engineered features improved the model's ability to identify suspicious claims.

---

# 🤖 Models Evaluated

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Linear SVM
- RBF SVM
- PCA + SVM

---

# 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------:|----------:|--------:|---------:|---------:|
| Logistic Regression | **84%** | **77%** | **49%** | **60%** | **84%** |
| Linear SVM | 83% | 73% | 49% | 59% | 81% |
| KNN | 74% | 42% | 16% | 24% | 67% |
| Naive Bayes | 62% | 33% | **55%** | 42% | 64% |
| RBF SVM | 76% | 60% | 6% | 11% | 85% |
| PCA + SVM | 77% | 80% | 8% | 15% | 82% |

### 🏆 Best Model

**Logistic Regression**

Selected because it achieved the best balance between:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# 🔄 Machine Learning Workflow

```text
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Engineering
    │
    ▼
Preprocessing Pipeline
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Model Comparison
    │
    ▼
Streamlit Deployment
```

---

# 🖥️ Streamlit Dashboard

The application contains the following pages:

### 🏠 Home

- Modern Landing Page
- Professional UI
- Project Overview

### 🔍 Predict Fraud

- Insurance Claim Form
- Real-Time Prediction
- Fraud Probability
- Risk Level
- Confidence Score

### 📊 Analytics

- Fraud Distribution
- Claim Analysis
- Incident Severity Analysis
- Interactive Plotly Charts

### 📈 Model Comparison

- Performance Comparison
- Evaluation Metrics
- Best Model Selection

### ℹ️ About

- Business Problem
- Dataset Information
- Workflow
- Project Summary

---

# ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Car-Insurance-Fraud-Detection.git
```

Navigate to the project folder:

```bash
cd Car-Insurance-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 📸 Application Screenshots

> Add screenshots inside the `screenshots/` folder and update the paths below.

### Home Page

```text
screenshots/home.png
```

### Prediction Page

```text
screenshots/prediction.png
```

### Analytics Page

```text
screenshots/analytics.png
```

### Model Comparison

```text
screenshots/comparison.png
```

---

# 🎯 Future Improvements

- SHAP Explainable AI
- XGBoost & LightGBM Comparison
- Cloud Deployment
- User Authentication
- Claim History Database
- PDF Report Generation

---

# 👨‍💻 Author

**Kunal Walunj**

B.Tech Student | Machine Learning Enthusiast

---

## ⭐ If you found this project useful, consider giving it a Star!
