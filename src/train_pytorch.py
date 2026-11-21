import copy

import pandas as pd
import torch
from torch import nn

from preprocess import prepare_data, split_data, build_preprocessor


class ChurnMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def prepare_tensors(X_train, X_val, y_train, y_val):
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    X_train_tensor = torch.tensor(
        X_train_processed,
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        X_val_processed,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    y_val_tensor = torch.tensor(
        y_val.to_numpy(),
        dtype=torch.float32,
    ).unsqueeze(1)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor


def train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 500
    patience = 20

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()

        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor)

        # Track best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"Train loss = {loss.item():.4f}, "
                f"Val loss = {val_loss.item():.4f}"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model weights
    model.load_state_dict(best_model_state)
    model.eval()

    print(f"Best validation loss: {best_val_loss:.4f}")

    return model


def main():
    # Load data
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X, y = prepare_data(df)

    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    # Prepare tensors
    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = (
        prepare_tensors(X_train, X_val, y_train, y_val)
    )

    # Initialize model
    torch.manual_seed(42)

    model = ChurnMLP(input_size=X_train_tensor.shape[1])

    # Train model
    model = train_model(
        model,
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
    )


if __name__ == "__main__":
    main()
