from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from evaluate import evaluate_model, print_metrics
from preprocess import prepare_data, split_data
from train_sklearn import build_logistic_pipeline


def main():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X, y = prepare_data(df)

    X_train, _, X_test, y_train, _, y_test = split_data(X, y)

    threshold = 0.3

    model = build_logistic_pipeline(X_train)
    model.fit(X_train, y_train)

    model_path = Path("models/churn_pipeline.joblib")

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_model(y_test, y_pred, y_proba)

    print_metrics("Final Logistic Regression — Test", metrics)

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
