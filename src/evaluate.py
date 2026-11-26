import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocess import prepare_data, split_data
from train_sklearn import (
    build_dummy_pipeline,
    build_logistic_pipeline,
    build_random_forest_pipeline,
)


def evaluate_model(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }


def print_metrics(model_name, metrics):
    print(f"\n{model_name}")

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


def main():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X, y = prepare_data(df)
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    models = {
        "Dummy": build_dummy_pipeline(X_train),
        "Logistic Regression": build_logistic_pipeline(X_train),
        "Random Forest": build_random_forest_pipeline(X_train),
    }

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = evaluate_model(y_val, y_pred, y_proba)
        print_metrics(model_name, metrics)

        if model_name == "Logistic Regression":
            for threshold in [0.3, 0.5, 0.7]:
                threshold_pred = (y_proba >= threshold).astype(int)

                print(f"\nThreshold: {threshold}")
                print(
                    f"Precision: "
                    f"{precision_score(y_val, threshold_pred, zero_division=0):.3f}"
                )
                print(
                    f"Recall: "
                    f"{recall_score(y_val, threshold_pred, zero_division=0):.3f}"
                )
                print(
                    f"F1-score: {f1_score(y_val, threshold_pred, zero_division=0):.3f}"
                )

            threshold_pred = (y_proba >= 0.3).astype(int)

            cm = confusion_matrix(y_val, threshold_pred)

            print("\nLogistic Regression confusion matrix (threshold=0.3):")
            print(cm)


if __name__ == "__main__":
    main()
