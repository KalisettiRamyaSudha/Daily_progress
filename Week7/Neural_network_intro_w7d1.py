"""
Week 7 Day 1 - Neural Network Introduction
Review the basics of artificial neural networks and experiment with a
simple MLP (Multi-Layer Perceptron) classifier.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

print(
    "A neural network, in short: layers of simple units ('neurons'), each "
    "computing a weighted sum of its inputs plus a bias, passed through a "
    "non-linear 'activation function' (like ReLU). Stacking layers lets "
    "the network learn non-linear patterns that a single linear model "
    "(like LogisticRegression) can't. Training (backpropagation) nudges "
    "every weight to reduce prediction error, repeated over many passes "
    "through the data ('epochs')."
)

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Neural networks are sensitive to feature scale, like KNN/KMeans/PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# A simple MLP: one hidden layer of 10 neurons. sklearn's MLPClassifier
# handles the architecture, activation (default 'relu'), and training loop
# for us.
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

predictions = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"\nMLP accuracy on Iris test set: {accuracy:.4f}")
print(f"Network structure: {X.shape[1]} inputs -> 10 hidden neurons -> {len(set(y))} outputs")
print(f"Training stopped after {mlp.n_iter_} epochs (max_iter=1000)")
print(f"Final training loss: {mlp.loss_:.4f}")