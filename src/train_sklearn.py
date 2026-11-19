from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from preprocess import build_preprocessor

def build_dummy_pipeline(X_train):
    pipeline = Pipeline(
        steps = [
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", DummyClassifier(strategy = 'most_frequent'))
        ]
    )

    return pipeline

