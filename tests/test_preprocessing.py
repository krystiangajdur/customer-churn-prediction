import pandas as pd

from preprocess import prepare_data, split_data

def test_prepare_data():
    df = pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "tenure": [0, 12, 24],
        "TotalCharges": [" ", "120.5", "300.0"],
        "Churn": ["No", "Yes", "No"],
    })

    X, y = prepare_data(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert X["TotalCharges"].tolist() == [0.0, 120.5, 300.0]
    assert y.tolist() == [0, 1, 0]


def test_prepare_data_does_not_modify_original():
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "tenure": [0, 12],
        "TotalCharges": [" ", "120.5"],
        "Churn": ["No", "Yes"],
    })

    original_df = df.copy(deep=True)

    prepare_data(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_split_data():
    X = pd.DataFrame({
        "feature": range(100),
    })

    y = pd.Series([0] * 75 + [1] * 25)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20

    assert len(y_train) == 60
    assert len(y_val) == 20
    assert len(y_test) == 20

    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)