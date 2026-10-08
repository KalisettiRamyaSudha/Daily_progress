"""
Week 5 Day 3 - Cross Validation Basics
Use cross_val_score to assess model reliability rather than trusting a
single train/test split.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# cross_val_score splits the data into 5 folds, trains on 4 and tests on the
# 5th, rotating which fold is held out each time - so every row gets used
# for both training and testing across the 5 runs. This gives 5 accuracy
# scores instead of just one, which shows how much performance depends on
# which particular rows ended up in the test set.
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name}:")
    print(f"  Fold scores: {np.round(scores, 4)}")
    print(f"  Mean accuracy: {scores.mean():.4f}")
    print(f"  Std deviation:  {scores.std():.4f}  <- how much scores vary across folds")
    print()

print(
    "A single train/test split only tells you how the model did on one "
    "particular 20% of the data - it could be a lucky or unlucky split. "
    "Cross-validation's mean and std dev across 5 different splits gives a "
    "much more reliable estimate of how the model will perform on unseen data."
)