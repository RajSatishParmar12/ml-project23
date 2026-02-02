import joblib
from sklearn.datasets import load_iris

# Load model
model = joblib.load("model.pkl")

# Load test data
X, y = load_iris(return_X_y=True)

# Make prediction
pred = model.predict(X)

# Simple assertion test
assert len(pred) == len(y)

print("Test passed successfully.")
