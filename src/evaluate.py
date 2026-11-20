import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

from preprocess import prepare_data, split_data, build_preprocessor
from train_sklearn import build_dummy_pipeline, build_logistic_pipeline, build_random_forest_pipeline

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

X, y = prepare_data(df)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

dummy_pipeline = build_dummy_pipeline(X_train)
dummy_pipeline.fit(X_train, y_train)
y_pred_dum = dummy_pipeline.predict(X_val)
y_dum_prob = dummy_pipeline.predict_proba(X_val)[:, 1]

print("Dummy predictions:", y_pred_dum[:10])
print("Accuracy:", round(accuracy_score(y_val, y_pred_dum), 3))
print("Precision:", round(precision_score(y_val, y_pred_dum, zero_division = 0), 3))
print("Recall:", round(recall_score(y_val, y_pred_dum), 3))
print("F1-score:", round(f1_score(y_val, y_pred_dum), 3))
print("ROC-AUC:", round(roc_auc_score(y_val, y_dum_prob), 3))
print("Average Precision:", round(average_precision_score(y_val, y_dum_prob), 3))


logistic_pipeline = build_logistic_pipeline(X_train)
logistic_pipeline.fit(X_train, y_train)
y_pred_log = logistic_pipeline.predict(X_val)
y_log_prob = logistic_pipeline.predict_proba(X_val)[:, 1]

print("\nLogistic regression Predictions:", y_pred_log[:10])
print("Accuracy:", round(accuracy_score(y_val, y_pred_log), 3))
print("Precision:", round(precision_score(y_val, y_pred_log), 3))
print("Recall:", round(recall_score(y_val, y_pred_log), 3))
print("F1-score:", round(f1_score(y_val, y_pred_log), 3))
print("ROC-AUC:", round(roc_auc_score(y_val, y_log_prob), 3))
print("Average Precision:", round(average_precision_score(y_val, y_log_prob), 3))
for threshold in [0.3, 0.5, 0.7]:
    y_pred = (y_log_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision_score(y_val, y_pred):.3f}")
    print(f"Recall: {recall_score(y_val, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_val, y_pred):.3f}")

threshold = 0.3

y_proba = logistic_pipeline.predict_proba(X_val)[:, 1]
y_pred = (y_proba >= threshold).astype(int)

cm = confusion_matrix(y_val, y_pred)

print(cm)


rf_pipeline = build_random_forest_pipeline(X_train)
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_val)
y_rf_prob = rf_pipeline.predict_proba(X_val)[:, 1]

print("\nRandom forest predictions:", y_pred_rf[:10])
print("Accuracy:", round(accuracy_score(y_val, y_pred_rf), 3))
print("Precision:", round(precision_score(y_val, y_pred_rf), 3))
print("Recall:", round(recall_score(y_val, y_pred_rf), 3))
print("F1-score:", round(f1_score(y_val, y_pred_rf), 3))
print("ROC-AUC:", round(roc_auc_score(y_val, y_rf_prob), 3))
print("Average Precision:", round(average_precision_score(y_val, y_rf_prob), 3))
