
import numpy as np
import torch
import lightning as L
from sklearn.metrics import mean_squared_error

from lstm_model import LightningLSTM


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
