
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam


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
