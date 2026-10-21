"""
Week 7 Day 1 - Perceptron or MLP Basics
Build a basic MLP model on a small classification dataset (Breast Cancer,
for continuity with earlier weeks).
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# A basic MLP: one hidden layer of 16 neurons, trained with the default
# 'adam' optimizer and 'relu' activation.
mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

predictions = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"MLP Accuracy: {accuracy:.4f}")
print(f"Hidden layers: {mlp.hidden_layer_sizes}")
print(f"Activation function: {mlp.activation}")
print(f"Number of weight layers (input->hidden, hidden->output): {len(mlp.coefs_)}")
for i, coef in enumerate(mlp.coefs_):
    print(f"  Layer {i+1} weight matrix shape: {coef.shape}")
print(f"\nClassification Report:")
print(classification_report(y_test, predictions, target_names=cancer.target_names))