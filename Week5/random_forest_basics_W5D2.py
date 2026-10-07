"""
Week 5 Day 2 - Random Forest Basics
Build and evaluate a random forest classifier.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# A random forest trains many decision trees on random subsets of the data
# and features, then averages their votes - this usually generalizes better
# than any single tree.
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

predictions = forest_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Random Forest Accuracy: {accuracy:.4f}")
print(f"Number of trees: {forest_model.n_estimators}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=cancer.target_names))