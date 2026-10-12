# Daily Progress — Python, Data Analysis & Machine Learning

A concise five-week log: Python fundamentals through classification, ensembles, and model tuning.

## Week 1 — Python Fundamentals
Conditionals, loops, functions, string manipulation, and file I/O — building up to a combined practice set and basic Git organization.

## Week 2 — NumPy & pandas
Array creation and math with NumPy (indexing, reshaping, random data, stats); DataFrame creation, cleaning, and `groupby` aggregation with pandas — capped with a mini dataset project.

## Week 3 — Data Visualization & EDA
Matplotlib and Seaborn chart types (line, bar, histogram, scatter, boxplot, heatmap); data cleaning, outlier detection (IQR), and dtype conversion — capped with a mini EDA project on a messy employee dataset (6 charts, 5 documented findings).

## Week 4 — Machine Learning Basics
The core `fit → predict → evaluate` workflow: train/test splitting, Linear & Logistic Regression, KNN, feature scaling, and evaluation metrics (MAE/RMSE/R², accuracy/precision/recall/F1, confusion matrix) — capped with a mini ML project classifying wine cultivars (97.2% accuracy, two models compared).

## Week 5 — Trees, Ensembles & Model Tuning
Decision trees and overfitting via `max_depth`; Random Forests and why ensembling reduces variance; cross-validation (`cross_val_score`) and hyperparameter tuning (`GridSearchCV`); full preprocessing pipelines (imputing, encoding, scaling via `ColumnTransformer`) — capped with a mini project predicting student pass/fail outcomes (Logistic Regression edged out Random Forest, F1 0.769 vs. 0.750).

## Key Takeaways
- **Fundamentals → data handling:** control flow and functions gave way to vectorized NumPy/pandas operations as the default way to work with data.
- **EDA discipline:** always inspect, clean, and visualize before modeling — and back every finding with an actual number.
- **The ML workflow is universal:** split → preprocess (fit on train only) → train → predict → evaluate with more than one metric, whether the model is linear, KNN, a tree, or a forest.
- **Model selection needs evidence:** cross-validation and multi-metric comparison (not one accuracy number from one split) are what justify picking one model over another.

## Pushing to Git
```bash
cd Daily_progress
cp README.md .   # replaces the repo's root README with this summary
git add README.md
git commit -m "Update summary README through Week 5"
git push origin main
```