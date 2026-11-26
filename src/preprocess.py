import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    y = df["Churn"].map({"No": 0, "Yes": 1})

    X = df.drop(columns=["customerID", "Churn"])

    return X, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X_train):
    numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns

    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), numerical_features),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return preprocessor
