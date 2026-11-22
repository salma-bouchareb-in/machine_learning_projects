Spam/Ham Email Classifier

A machine learning project that classifies emails as **spam** or **ham** (not spam) using TF-IDF and three classification models.

## Overview

- **Goal:** detect spam emails automatically.
- **Text encoding:** TF-IDF (`stop_words="english"`, `max_features=5000`, `ngram_range=(1,2)`).
- **Models compared:** Logistic Regression, LinearSVC, K-Nearest Neighbors (KNN).
- **Tuning:** `RandomizedSearchCV` with 5-fold cross-validation, scored on F1.
- **Class imbalance:** handled with `class_weight="balanced"` and a stratified train/test split.

## Dataset

- File: `spam.csv`
- Columns: `Category` (ham/spam) and `Message` (email text)
- Size: ... emails (... % spam)

## Project Steps

1. Load and explore the data
2. Encode labels (`ham` = 0, `spam` = 1)
3. Split into train (80%) and test (20%) with stratification
4. Convert text to numbers with TF-IDF
5. Tune and train each model with `RandomizedSearchCV`
6. Evaluate with accuracy, precision, recall and F1 score
7. Compare the models and test on new emails

## Results

| Model | Best Parameters | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Logistic Regression | C=10 | 0.981 | 0.950 | 0.891 | 0.919 |
| LinearSVC | C=1 | 0.981 | 0.966 | 0.875 | 0.918 |
| KNN | n_neighbors=3 | 0.921 | 1.000 | 0.359 | 0.529 |

**Best model:
  ** Logistic Regression (F1 = 0.919)
Logistic Regression and LinearSVC perform almost identically (F1 ≈ 0.92).
KNN has perfect precision but misses about 64% of spam emails (recall = 0.36),
because distance-based methods struggle with high-dimensional, sparse TF-IDF data.

## Technologies

- Python
- pandas, NumPy
- scikit-learn
- matplotlib
- Jupyter Notebook

## How to Run

```bash
git clone https://github.com/salma-bouchareb-in/machine_learning_projects
cd machine-learning-projects/spam_or_ham
pip install -r requirements.txt
jupyter notebook spam.ipynb
```

## Test the Model on a New Email

```python
email = "Congratulations! You won a free prize, click here now"
pred = best_models["LinearSVC"].predict(tfidf.transform([email]))[0]
print("SPAM" if pred == 1 else "HAM")
```

## Author

Salma Bouchareb
[GitHub](https://github.com/salma-bouchareb-in)
