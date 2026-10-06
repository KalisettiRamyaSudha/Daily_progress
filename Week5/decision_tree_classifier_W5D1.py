
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

predictions = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Decision Tree Accuracy: {accuracy:.4f}")
print(f"Tree depth used: {tree_model.get_depth()}")
print(f"Number of leaves: {tree_model.get_n_leaves()}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=cancer.target_names))