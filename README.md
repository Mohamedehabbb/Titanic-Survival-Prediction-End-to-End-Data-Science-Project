# Titanic-Survival-Prediction-End-to-End-Data-Science-Project
data science project
# 🚢 Titanic Survival Prediction – End‑to‑End Data Science Project

## 📌 Project Overview

The sinking of the Titanic is one of the most well‑known tragedies in history. In this project, we tackle a classic supervised machine learning problem: **predicting passenger survival** based on demographic and socio‑economic features such as gender, age, ticket class, and family relationships onboard.

This repository presents a **complete, real‑world Data Science workflow**, starting from raw CSV files and ending with model comparison, interpretation, and actionable insights. The focus is not only on model performance, but also on **clear reasoning, exploratory analysis, and systematic improvement**.

---

## 🎯 Problem Statement

Given passenger information, the goal is to predict whether a passenger **survived (1)** or **did not survive (0)** the Titanic disaster.

This is a **binary classification problem**, where accuracy and model interpretability are key evaluation criteria.

---

## 📂 Dataset Description

The dataset is provided by Kaggle and consists of three CSV files:

* **train.csv** – Training data including the target variable `Survived`
* **test.csv** – Test data without survival labels (used for final predictions)
* **gender_submission.csv** – Example submission format

### Target Variable

* `Survived`: 0 = Did not survive, 1 = Survived

### Key Features

* `Pclass`: Passenger class (proxy for socio‑economic status)
* `Sex`: Passenger gender
* `Age`: Age in years
* `SibSp`: Number of siblings/spouses aboard
* `Parch`: Number of parents/children aboard
* `Fare`: Ticket fare
* `Embarked`: Port of embarkation

---

## 🧠 Project Workflow & Methodology

This project follows a structured Data Science pipeline:

### 1️⃣ Data Understanding & Initial Inspection

* Loaded raw CSV files
* Inspected data types, distributions, and missing values
* Identified key data quality issues (missing `Age`, `Cabin`, `Embarked`)

---

### 2️⃣ Exploratory Data Analysis (EDA)

EDA was performed to uncover patterns and relationships between features and survival:

* **Survival distribution** to understand class balance
* **Gender vs Survival** revealing significantly higher survival rates for females
* **Passenger Class vs Survival** showing strong socio‑economic influence
* **Age distribution** analysis
* **Fare vs Survival** comparison

📊 Visualizations were used extensively to support data‑driven decisions.

---

### 3️⃣ Data Cleaning & Preprocessing

Key preprocessing steps included:

* Handling missing values:

  * `Age` imputed using the median
  * `Embarked` filled with the most frequent category
* Removing low‑information features:

  * Dropped `Cabin`, `Name`, and `Ticket`
* Ensured consistent preprocessing for both training and test datasets

---

### 4️⃣ Feature Engineering

To enhance model performance while preserving original column names, new features were introduced:

* **FamilySize** = `SibSp + Parch + 1`

This feature captures family presence onboard, which has a strong behavioral impact on survival probability.

---

### 5️⃣ Encoding & Scaling

* Categorical variables (`Sex`, `Embarked`) were encoded using one‑hot encoding
* Numerical features were standardized using `StandardScaler` for models sensitive to feature scale

---

### 6️⃣ Model Training

Multiple machine learning models were trained and evaluated to ensure robust comparison:

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Naive Bayes

Each model was trained using the same data split to ensure fair evaluation.

---

### 7️⃣ Model Evaluation

Models were evaluated using:

* Accuracy score
* Confusion matrix
* Precision, recall, and F1‑score

A comparative bar chart was used to visually summarize model performance.

---

### 8️⃣ Ensemble Learning

To further improve stability and performance, a **Voting Classifier** was implemented by combining:

* Logistic Regression
* SVM
* Random Forest

The ensemble model leveraged the strengths of both linear and tree‑based approaches, resulting in more balanced predictions.

---

## 📈 Key Results & Insights

* Gender and passenger class are the strongest predictors of survival
* Simple linear models performed competitively when paired with proper preprocessing
* Ensemble learning improved robustness and reduced model variance
* Feature engineering had a more noticeable impact than increasing model complexity

---

## ✅ Final Conclusion

This project demonstrates how a well‑structured Data Science approach can effectively solve a real‑world classification problem. By combining thoughtful EDA, disciplined preprocessing, and systematic model comparison, we achieved strong predictive performance while maintaining interpretability.

The notebook reflects industry‑level best practices and is suitable for:

* Data Science portfolios
* Technical interviews
* Educational reference

---

## 🚀 Future Work

Potential improvements to further enhance this project include:

* Hyperparameter tuning using GridSearchCV
* Advanced feature engineering (e.g., title extraction from names)
* Handling class imbalance with resampling techniques
* ROC‑AUC and Precision‑Recall analysis
* Deploying the model as a web application

---

## 📌 Author
Mohamed Ehab
Data Scientist | Machine Learning Enthusiast

Mohamed Ehab
Data Scientist | Machine Learning Enthusiast
