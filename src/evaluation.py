
import numpy as np
import torch
import lightning as L
from sklearn.metrics import mean_squared_error

from lstm_model import LightningLSTM

from torch.utils.data import Dataset, DataLoader
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_length]
        y_target = self.y[idx + self.seq_length]
        return X_seq, y_target


def make_dataloaders(X_train, X_test, y_train, y_test, seq_length, batch_size):
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_length)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        return None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader

def fit_model(num_features, seq_length, hidden_size, dropout, lr,
              train_loader, num_layers=1, max_epochs=100):
    model = LightningLSTM(
        num_features=num_features,
        seq_length=seq_length,
        hidden_size=hidden_size,
        dropout=dropout,
        lr=lr,
        num_layers=num_layers
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )

    trainer.fit(model, train_dataloaders=train_loader)
    return model


def get_predictions_log(dataloader, model, scaler_y):
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            predictions = np.append(predictions, y_pred.numpy().reshape(-1))
            actuals = np.append(actuals, y_batch.numpy().reshape(-1))

    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

    return np.expm1(predictions), np.expm1(actuals)


def compute_metrics(model, train_loader, test_loader, scaler_y):
    train_preds, train_actuals = get_predictions_log(train_loader, model, scaler_y)
    test_preds, test_actuals = get_predictions_log(test_loader, model, scaler_y)

    train_rmse = np.sqrt(mean_squared_error(train_actuals, train_preds))
    test_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "rmse_gap": test_rmse - train_rmse,
        "test_pred_std": np.std(test_preds, ddof=1),
        "test_actual_std": np.std(test_actuals, ddof=1),
        "test_std_ratio": np.std(test_preds, ddof=1) / np.std(test_actuals, ddof=1)
    }
