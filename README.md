# Credit Scoring – EDA & Interpretable Machine Learning

## Project Overview

This project performs an in-depth Exploratory Data Analysis (EDA) and develops interpretable machine learning models for credit scoring. The goal is to predict whether a client is creditworthy (1) or not creditworthy (0) using demographic, financial, and socio-economic features.

The workflow includes data cleaning, statistical analysis, feature engineering, class imbalance handling, and comparison of multiple interpretable models: Logistic Regression, Naive Bayes, and Decision Tree.

The main focus is model transparency and business interpretability rather than black-box performance.


## Business Objective

Target variable:
- 1 → Creditworthy client
- 0 → Not creditworthy client

From a financial risk perspective:
- False Positives (approving a non-creditworthy client) are highly costly.
- False Negatives (rejecting a creditworthy client) represent missed opportunities.

For this reason, precision is prioritized while maintaining a solid F1 score.


## Dataset

- 338,427 observations
- 19 features
- ID column removed during preprocessing

Feature types:

Binary:
- CODE_GENDER
- FLAG_OWN_CAR
- FLAG_OWN_REALTY
- FLAG_MOBIL
- FLAG_PHONE
- FLAG_WORK_PHONE
- FLAG_EMAIL

Categorical:
- NAME_INCOME_TYPE
- NAME_FAMILY_STATUS
- NAME_HOUSING_TYPE
- OCCUPATION_TYPE

Ordinal:
- NAME_EDUCATION_TYPE

Quantitative:
- CNT_CHILDREN
- DAYS_BIRTH
- DAYS_EMPLOYED
- CNT_FAM_MEMBERS
- AMT_INCOME_TOTAL

Class distribution:
- ~91% Non-creditworthy
- ~9% Creditworthy

The dataset is highly imbalanced.


## Data Cleaning & Feature Engineering

- Duplicate IDs checked (none found).
- ID column removed.
- Missing values in OCCUPATION_TYPE handled:
  - "Retired" if income type = Pensioner
  - "Missing" otherwise
- One corrupted observation removed.
- DAYS_BIRTH converted to AGE (years).
- DAYS_EMPLOYED converted to WORK_DURATION (years).
- FLAG_MOBIL removed (constant feature).
- CNT_CHILDREN removed due to multicollinearity with CNT_FAM_MEMBERS.
- Label encoding for binary and ordinal variables.
- One-hot encoding for nominal variables.
- MinMax normalization for quantitative features.
- Train-test split: 80% / 20%, stratified.


## Exploratory Data Analysis

Main findings:

- Income distribution is positively skewed.
- Creditworthy clients are mostly aged 43–51.
- Stable employment strongly correlates with creditworthiness.
- Most creditworthy clients live in two-member households (mainly couples without children).
- OCCUPATION_TYPE and NAME_INCOME_TYPE are strongly associated with TARGET.

Statistical tools used:
- Correlation heatmaps
- Chi-square tests
- Cramer's V
- T-tests
- KDE plots
- Boxplots


## Models Implemented

### 1. Logistic Regression (Baseline)
- F1 ≈ 19%
- Precision ≈ 46%
Strong bias toward majority class.

### 2. Logistic Regression (class_weight="balanced")
- F1 ≈ 45%
- Precision ≈ 31%
Improved balance but still limited performance.

### 3. Naive Bayes (with SMOTE + undersampling)
Models tested:
- GaussianNB
- CategoricalNB
- Hybrid Gaussian + Categorical

Strong overfitting observed. Test F1 remained below 31%.

### 4. Decision Tree (Best Model)

Hyperparameters tuned (controlled depth and splits).

Test performance:
- F1 ≈ 78%
- Precision ≈ 65%

The Decision Tree outperformed all other models while remaining fully interpretable.


## Model Interpretability

The final Decision Tree was visualized to analyze decision paths.

Key insight:
- WORK_DURATION is the most influential feature.
- Short employment duration strongly predicts non-creditworthiness.
- Income and occupation type significantly impact classification.

A manual walkthrough of a single prediction demonstrates alignment between business logic and model output.


## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Imbalanced-learn


## Conclusion

This project demonstrates how rigorous EDA combined with interpretable machine learning can deliver strong performance in credit scoring tasks.

While Logistic Regression and Naive Bayes struggled with imbalance and generalization, a tuned Decision Tree achieved high performance while preserving transparency and business interpretability.
