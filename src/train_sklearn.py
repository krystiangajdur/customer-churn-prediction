from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import build_preprocessor

def build_dummy_pipeline(X_train):
    pipeline = Pipeline(
        steps = [
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", DummyClassifier(strategy = 'most_frequent'))
        ]
    )
    return pipeline

def build_logistic_pipeline(X_train):
    pipeline = Pipeline(
        steps = [
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter = 1000, random_state = 42))
        ]
    )
    return pipeline

def build_random_forest_pipeline(X_train):
    pipeline = Pipeline(
        steps= [
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(n_estimators = 100, random_state = 42))
        ]
    )
    return pipeline

