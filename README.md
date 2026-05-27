# Identifying Player Engagement Through Regression Models
### Steam Games Dataset — Comparative Analysis

**By Robert Bamba and Shifra Abigail Garcia**

---

## Overview

This project applies multiple machine learning approaches to the Steam games dataset to understand what drives player engagement, measured by **average playtime forever**. Three different notebooks frame the same prediction problem in different ways — regression, classification, and class-balanced classification — to explore how problem framing and preprocessing choices affect model performance.

---

## Repository Structure

| Notebook | Task Type | Models Used |
|---|---|---|
| `Linear_Regression.ipynb` | Regression (continuous target) | Linear Regression, Ridge (L2), Lasso (L1) |
| `Logistic_Regression_SVM_Steam_Games.ipynb` | Classification (binary target) | Logistic Regression (L1/L2), LinearSVC (L1/L2) |
| `Logistic_Regression_SMOTE_Steam_Games.ipynb` | Classification with class balancing | Logistic Regression (L1/L2) + SMOTE |
| `Player_Engagement_Regression_Models.ipynb` | Unified comparative analysis | All of the above |
| `0_Player_Engagement_Regression_Models.ipynb` | Entry point / overview notebook | All of the above |

---

## Dataset

**Source:** [Steam Games Dataset — Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?utm_source=chatgpt.com&select=games_march2025_full.csv) (`games_steam.csv`)  
**Shape:** ~94,954 rows × 44 columns  
**Target Variable:** `average_playtime_forever`

Key features used across models include game metadata such as supported platforms (Windows, Mac, Linux), supported languages, genres, categories, developers, and publishers.

---

## Why Three Approaches?

All three notebooks share the same dataset and target variable but frame the problem differently:

- **Linear Regression** treats playtime as a *continuous* variable and asks: *how much* will a player engage?
- **Logistic Regression + SVM** binarizes playtime at the median and asks: *will a player engage highly or not?*
- **Logistic Regression + SMOTE** addresses the class imbalance that arises from that binarization.

These different framings — and the preprocessing choices that follow — explain the striking differences in performance metrics across notebooks.

---

## Methodology

### Data Cleaning (shared across all notebooks)
- Remove unnamed index columns from CSV export
- Strip extra whitespace from column names and string values
- Drop columns with more than 20,000 missing values (too sparse to be useful)
- Drop irrelevant metadata columns (screenshots, URLs, descriptions, review stats, etc.)
- Fill remaining numeric nulls with column medians
- Drop any rows still containing nulls after imputation

### Feature Engineering
- Multi-label encoding via `MultiLabelBinarizer` for genres, categories, supported languages, developers, and publishers
- Boolean conversion for platform support (Windows, Mac, Linux)
- Min-Max scaling for numeric features

### Models & Regularization
- **Linear Regression:** Baseline, Ridge (L2), Lasso (L1) with varying alpha values
- **Logistic Regression:** L1 and L2 regularization with hyperparameter tuning (`C`)
- **LinearSVC:** L1 and L2 regularization
- **SMOTE:** Synthetic Minority Over-sampling Technique to balance binary classes before training

### Evaluation Metrics
- **Regression:** R² score, MAE, MSE
- **Classification:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC-AUC

---

## Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn mglearn
```

---

## How to Run

1. Clone the repository and place `games_steam.csv` in the root directory.
2. Install dependencies (see above).
3. Open any notebook in Jupyter and run all cells top to bottom.
   - Start with `0_Player_Engagement_Regression_Models.ipynb` or `Player_Engagement_Regression_Models.ipynb` for a unified overview.
   - Open individual notebooks for deeper dives into each modeling approach.