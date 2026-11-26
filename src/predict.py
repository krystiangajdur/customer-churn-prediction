from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/churn_pipeline.joblib")
THRESHOLD = 0.3


def predict_churn(df):
    model = joblib.load(MODEL_PATH)

    X = df.copy()

    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

    X = X.drop(columns=["customerID", "Churn"], errors="ignore")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "churn_prediction": predictions,
        }
    )


def main():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    sample = df.head(5)

    results = predict_churn(sample)

    print(results)


if __name__ == "__main__":
    main()
