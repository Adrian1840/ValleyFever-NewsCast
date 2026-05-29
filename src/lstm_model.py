
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam

def prep_county_data_gen(PATH, county, training_percent=0.85):

    # Read aggregate data
    df = pd.read_csv(PATH + "AggregateData/" + county + "_Aggregate.csv")

    # County-specific columns to drop
    drop_cols = ["WIND_EventCount"]

    if county.lower() == "kern":
        drop_cols.append("FIRE_Acres_Burned")

    # Drop columns
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Merge news features
    df = df.merge(news_monthly, on=["Year-Month"], how="left")
    df = df.merge(article_features, on=["Year-Month"], how="left")

    # Fill NA values
    df = df.fillna(0)

    # Define predictors
    X = df.iloc[:, 2:]

    # Define response (logged case rates)
    y = np.log1p(df.iloc[:, 1:2])

    # Train/test split
    train_size = int(training_percent * len(X))
    test_size = len(X) - train_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

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
