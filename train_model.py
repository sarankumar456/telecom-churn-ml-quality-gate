import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_dataset(n=600, random_state=42):
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 73, n)
    monthly_charges = rng.uniform(25, 120, n).round(2)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n,
        p=[0.60, 0.25, 0.15]
    )

    tech_support = rng.choice(
        ["Yes", "No"], n, p=[0.35, 0.65]
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        n,
        p=[0.35, 0.50, 0.15]
    )

    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer"],
        n
    )

    # Create a reproducible synthetic churn target.
    churn_score = (
        1.5 * (contract == "Month-to-month")
        + 0.9 * (tenure < 18)
        + 0.7 * (monthly_charges > 85)
        + 0.8 * (tech_support == "No")
        + 0.4 * (internet_service == "Fiber optic")
    )

    churn = (churn_score >= 2.4).astype(int)

    # Add a small amount of label noise.
    flip = rng.random(n) < 0.04
    churn = np.where(flip, 1 - churn, churn)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "tech_support": tech_support,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "churn": churn
    })


def main():
    print("Generating telecom customer dataset...")

    data = generate_dataset()

    data.to_csv("telecom_churn.csv", index=False)

    X = data.drop(columns=["churn"])
    y = data["churn"]

    numeric_features = ["tenure", "monthly_charges"]
    categorical_features = [
        "contract",
        "tech_support",
        "internet_service",
        "payment_method"
    ]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, "telecom_churn_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "model_name": "Logistic Regression",
        "dataset": "Synthetic Telecom Customer Churn"
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Training completed.")
    print("Model accuracy:", round(accuracy, 4))
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()
