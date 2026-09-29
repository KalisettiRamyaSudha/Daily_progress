from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
df = iris.frame

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Full dataset size:", len(df))
print("Training set size:", len(X_train), f"({len(X_train) / len(df):.0%})")
print("Testing set size:", len(X_test), f"({len(X_test) / len(df):.0%})")

print("\nClass balance in full dataset:\n", y.value_counts(normalize=True).round(2))
print("\nClass balance in training set:\n", y_train.value_counts(normalize=True).round(2))
print("\nClass balance in test set:\n", y_test.value_counts(normalize=True).round(2))
print("\n(stratify=y kept the class proportions consistent across all three)")