import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from lstm_model import LightningLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam

# Convert 2D input into LSTM-ready 3D tensor
def create_lstm_input(X, seq_length):
    X_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
    return torch.tensor(np.array(X_seq), dtype=torch.float32)

# Evaluate model on input and return RMSE (in original scale)
def evaluate_model(model, X_tensor, y_true_scaled, scaler_y, seq_length):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).squeeze().numpy()
    preds_original = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_original = scaler_y.inverse_transform(y_true_scaled[seq_length:]).flatten()
    return np.sqrt(mean_squared_error(y_original, preds_original))

# Compute Permutation Feature Importance
def compute_pfi(X_test, y_test, model, scaler_y, feature_names, seq_length):
    X_test_lstm = create_lstm_input(X_test, seq_length)
    baseline_rmse = evaluate_model(model, X_test_lstm, y_test, scaler_y, seq_length)

    pfi_scores = {}
    for i, col in enumerate(feature_names):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        X_perm_lstm = create_lstm_input(X_test_permuted, seq_length)
        perm_rmse = evaluate_model(model, X_perm_lstm, y_test, scaler_y, seq_length)
        pfi_scores[col] = perm_rmse - baseline_rmse

    return dict(sorted(pfi_scores.items(), key=lambda x: x[1], reverse=True))
    
def run_lstm_pfi_experiment(
    county_name,
    num_features,
    seq_length,
    hidden_size,
    dropout,
    lr,
    num_layers,
    train_loader,
    test_loader,
    X_test,
    y_test,
    scaler_y,
    feature_names,
    num_runs=20,
    max_epochs=100,
    save_csv=True
):

    all_pfi_runs = []

    for run in range(num_runs):

        print(f"{county_name} Run {run + 1}/{num_runs}")

        # --- Model ---
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

        # --- Compute PFI ---
        pfi_scores = compute_pfi(
            X_test=X_test,
            y_test=y_test,
            model=model,
            scaler_y=scaler_y,
            feature_names=feature_names,
            seq_length=seq_length
        )

        # Convert dict -> DataFrame
        pfi_scores = pd.DataFrame({
            "feature": list(pfi_scores.keys()),
            "pfi": list(pfi_scores.values())
        })

        pfi_scores["run"] = run + 1
        pfi_scores["county"] = county_name

        all_pfi_runs.append(pfi_scores)

    # --- Combine all runs ---
    pfi_df = pd.concat(all_pfi_runs, ignore_index=True)

    # --- Save ---
    if save_csv:
        filename = f"{county_name.lower()}_pfis_{num_runs}.csv"
        pfi_df.to_csv(filename, index=False)
        print(f"Saved: {filename}")

    return pfi_df
