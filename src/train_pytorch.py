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



df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

X, y = prepare_data(df)

X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = (
    prepare_tensors(X_train, X_val, y_train, y_val)
)

model = ChurnMLP(input_size=X_train_tensor.shape[1])

logits = model(X_train_tensor)

criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, y_train_tensor)

print("Training samples:", X_train_tensor.shape[0])
print("Validation samples:", X_val_tensor.shape[0])
print("Model input features:", X_train_tensor.shape[1])
print("Initial training loss:", loss.item())

