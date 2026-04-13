# Amazon Product Review Sentiment Analysis + ETL Pipeline

## Problem
Customer reviews contain valuable insights, but they are unstructured and difficult to analyze manually.  
This project aims to analyze customer sentiment (positive, neutral, negative) from Amazon product reviews and build an end-to-end ETL pipeline.

---

## Dataset
- Source: Kaggle - Amazon Product Reviews Dataset  
- Features used:
  - reviews.text
  - reviews.rating
  - brand
  - categories  

---

## Pipeline
This project follows an ETL + Machine Learning pipeline:

**Extract**
- Load dataset using Pandas

**Transform**
- Handle missing values
- Clean text (lowercase, remove symbols)
- Generate sentiment labels from rating

**Model**
- Convert text to numerical features using TF-IDF
- Train Logistic Regression model

**Load**
- Save prediction results to CSV

---

## Tools & Libraries
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Regex (re)

---

## Model
- Algorithm: Logistic Regression
- Feature Engineering: TF-IDF Vectorizer (max_features=5000)

---

## Results
- Classification Report:
  - Precision, Recall, F1-score per class
- Sentiment Distribution Visualization (Bar Chart)

Example insight:
- Majority of reviews are **positive**, indicating high customer satisfaction
- A notable portion of **negative reviews** suggests areas for product improvement

---

## Visualization
- Sentiment distribution using bar chart
- Helps identify overall customer satisfaction

---
