import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, pacf


def prep_county_sarimax(
    aggregate_path,
    news_features_path,
    county,
    training_percent=0.85
):
    """
    Prepare county-level Valley Fever data for SARIMAX modeling.
    """

    df = pd.read_csv(aggregate_path)
    news_features = pd.read_csv(news_features_path)

    drop_cols = ["WIND_EventCount"]

    if county.lower() == "kern":
        drop_cols.append("FIRE_Acres_Burned")

    df = df.drop(
        columns=[col for col in drop_cols if col in df.columns]
    )

    df = df.merge(
        news_features,
        on="Year-Month",
        how="left"
    )

    df = df.fillna(0)

    df["Year-Month"] = pd.to_datetime(df["Year-Month"])
    df = df.sort_values("Year-Month").set_index("Year-Month")
    df = df.asfreq("MS")

    y = np.log1p(df["VFRate"])
    X = df.drop(columns=["VFRate"])

    train_size = int(training_percent * len(df))
    test_size = len(df) - train_size

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    return df, X_train, X_test, y_train, y_test, train_size, test_size


def chronological_split(y, X, train_percent=0.85):
    """
    Split a time series and exogenous dataframe chronologically.
    """

    train_size = int(len(y) * train_percent)

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    return y_train, y_test, X_train, X_test


def fit_sarimax(
    y_train,
    X_train,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12)
):
    """
    Fit a SARIMAX model with exogenous predictors.
    """

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    return results


def evaluate_sarimax(results, y_train, y_test, X_train, X_test):
    """
    Generate train/test SARIMAX predictions and compute RMSE on original scale.
    """

    train_pred_log = results.predict(
        start=y_train.index[0],
        end=y_train.index[-1],
        exog=X_train
    )

    test_pred_log = results.predict(
        start=y_test.index[0],
        end=y_test.index[-1],
        exog=X_test
    )

    train_pred = np.expm1(train_pred_log)
    test_pred = np.expm1(test_pred_log)

    train_actual = np.expm1(y_train)
    test_actual = np.expm1(y_test)

    train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))

    return (
        train_pred,
        test_pred,
        train_actual,
        test_actual,
        train_rmse,
        test_rmse
    )


def check_stationarity(series, county_name="County"):
    """
    Run Augmented Dickey-Fuller stationarity test.
    """

    result = adfuller(series.dropna(), autolag="AIC")

    adf_stat = result[0]
    p_value = result[1]

    print(f"\n--- {county_name} Stationarity Test ---")
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Stationary")
    else:
        print("Result: Non-Stationary")

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "is_stationary": p_value < 0.05
    }
