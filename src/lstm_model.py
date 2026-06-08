
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prep_county_data_gen(
    aggregate_path,
    news_features_path,
    county,
    training_percent=0.85
):
    """
    Prepare county-level Valley Fever data for LSTM modeling.
    Returns
        df, X_train, X_test, y_train, y_test, scaler_X, scaler_y,
        train_size, test_size
    """

    # Read county aggregate and news feature data
    df = pd.read_csv(aggregate_path)
    news_features = pd.read_csv(news_features_path)

    # County-specific columns to drop
    drop_cols = ["WIND_EventCount"]

    if county.lower() == "kern":
        drop_cols.append("FIRE_Acres_Burned")

    df = df.drop(
        columns=[col for col in drop_cols if col in df.columns]
    )

    # Merge unlagged news features
    df = df.merge(
        news_features,
        on="Year-Month",
        how="left"
    )

    # Fill missing values
    df = df.fillna(0)

    # Predictors: drop Year-Month and VFRate
    X = df.drop(columns=["Year-Month", "VFRate"])

    # Response: log-transformed case rate
    y = np.log1p(df[["VFRate"]])

    # Train/test split
    train_size = int(training_percent * len(X))
    test_size = len(X) - train_size

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    return (
        df,
        X_train,
        X_test,
        y_train,
        y_test,
        scaler_X,
        scaler_y,
        train_size,
        test_size
    )


class LightningLSTM(L.LightningModule):
    def __init__(self, num_features, seq_length, hidden_size=16,
                 dropout=0.0, lr=0.001, num_layers=1):
        super().__init__()

        self.num_features = num_features
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.lr = lr
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout)
        self.predictor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(-1, self.seq_length, self.num_features)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        dropped = self.dropout(last_time_step)
        return self.predictor(dropped)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

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

