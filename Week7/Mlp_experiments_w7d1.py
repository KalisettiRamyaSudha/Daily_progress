"""
Week 7 Day 1 - Activation and Epoch Observation
Change epochs (max_iter) and hidden-layer settings and observe how
performance changes.
"""
import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_breast_cancer

# Low max_iter values are intentionally used below to show under-training -
# the resulting "hasn't converged yet" warning is expected, not a bug, so
# it's suppressed to keep the output focused on the actual comparison.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Experiment 1: vary epochs (max_iter) ---
print("=== Experiment 1: Varying epochs (max_iter) ===")
epoch_results = []
for max_iter in [5, 20, 50, 100, 300, 1000]:
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=max_iter, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_scaled))
    epoch_results.append({"max_iter": max_iter, "actual_epochs_run": mlp.n_iter_, "accuracy": acc, "final_loss": mlp.loss_})

epoch_df = pd.DataFrame(epoch_results).round(4)
print(epoch_df.to_string(index=False))

# --- Experiment 2: vary hidden layer size/depth ---
print("\n=== Experiment 2: Varying hidden-layer architecture ===")
architectures = {
    "No hidden layer (linear-ish, 1 tiny unit)": (1,),
    "Small: (4,)": (4,),
    "Medium: (16,)": (16,),
    "Large: (64,)": (64,),
    "Deep: (32, 16)": (32, 16),
    "Deep: (32, 16, 8)": (32, 16, 8),
}
arch_results = []
for name, layers in architectures.items():
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_scaled))
    arch_results.append({"architecture": name, "accuracy": acc, "epochs_run": mlp.n_iter_})

arch_df = pd.DataFrame(arch_results).round(4)
print(arch_df.to_string(index=False))

# --- Experiment 3: vary activation function ---
print("\n=== Experiment 3: Varying activation function ===")
activation_results = []
for activation in ["identity", "logistic", "tanh", "relu"]:
    mlp = MLPClassifier(hidden_layer_sizes=(16,), activation=activation, max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test_scaled))
    activation_results.append({"activation": activation, "accuracy": acc, "epochs_run": mlp.n_iter_})

activation_df = pd.DataFrame(activation_results).round(4)
print(activation_df.to_string(index=False))

print(
    "\nObservations:\n"
    "- Too few epochs (max_iter=5-20) cuts training off before the network "
    "converges - accuracy and loss are both worse than letting it run longer.\n"
    "- A single unit (no real hidden layer) acts almost like a linear "
    "model and underperforms networks with a real hidden layer.\n"
    "- Beyond a certain hidden-layer size, accuracy mostly plateaus on a "
    "dataset this size (569 rows) - bigger isn't automatically better, "
    "and a deeper network needs more data/epochs to earn its extra capacity.\n"
    "- 'identity' activation (no non-linearity at all) behaves like plain "
    "logistic regression; relu/tanh/logistic all introduce the "
    "non-linearity that lets the network fit more complex patterns."
)