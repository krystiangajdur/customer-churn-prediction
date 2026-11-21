import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from preprocess import prepare_data, split_data
from train_pytorch import ChurnMLP, prepare_tensors, train_model
from evaluate import evaluate_model, print_metrics


def main():
    # Load and split data
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X, y = prepare_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Prepare training and validation tensors
    (X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, preprocessor) = prepare_tensors(
        X_train, X_val, y_train, y_val, return_preprocessor=True)

    # Transform test data using the training preprocessor
    X_test_processed = preprocessor.transform(X_test)

    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    # Initialize and train MLP
    torch.manual_seed(42)

    model = ChurnMLP(input_size=X_train_tensor.shape[1])

    model = train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

    # Final test evaluation
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_probabilities = torch.sigmoid(test_logits)

    y_proba = test_probabilities.squeeze(1).numpy()
    y_true = y_test.to_numpy()

    # Threshold selected during validation
    threshold = 0.3
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate test metrics
    metrics = evaluate_model(y_true, y_pred, y_proba)

    print_metrics("PyTorch MLP — Test", metrics)

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
