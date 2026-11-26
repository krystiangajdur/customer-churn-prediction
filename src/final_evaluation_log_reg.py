import pandas as pd
from sklearn.metrics import confusion_matrix

from evaluate import evaluate_model, print_metrics
from preprocess import prepare_data, split_data
from train_sklearn import build_logistic_pipeline


def main():
    # Load and split data
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X, y = prepare_data(df)

    X_train, _, X_test, y_train, _, y_test = split_data(X, y)

    # Selected model and threshold from validation
    threshold = 0.3

    model = build_logistic_pipeline(X_train)
    model.fit(X_train, y_train)

    # Final test predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Evaluate on test set
    metrics = evaluate_model(y_test, y_pred, y_proba)

    print_metrics("Final Logistic Regression — Test", metrics)

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
