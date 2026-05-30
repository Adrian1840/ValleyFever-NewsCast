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


def plot_pacf_plotly(series, title="", lags=24, color="royalblue"):
    """
    Plot PACF values using Plotly with confidence interval shading.
    """

    clean_series = series.dropna()

    pacf_vals = pacf(
        clean_series,
        nlags=lags,
        method="ywm"
    )

    lag_vals = np.arange(len(pacf_vals))
    ci = 1.96 / np.sqrt(len(clean_series))

    fig = go.Figure()

    fig.add_hrect(
        y0=-ci,
        y1=ci,
        fillcolor="lightsteelblue",
        opacity=0.35,
        line_width=0,
        layer="below"
    )

    for lag, val in zip(lag_vals, pacf_vals):
        fig.add_trace(
            go.Scatter(
                x=[lag, lag],
                y=[0, val],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip",
                showlegend=False
            )
        )

    fig.add_trace(
        go.Scatter(
            x=lag_vals,
            y=pacf_vals,
            mode="markers",
            marker=dict(
                size=8,
                color=color,
                line=dict(color=color, width=1)
            ),
            showlegend=False
        )
    )

    fig.add_hline(
        y=0,
        line=dict(color="gray", width=1.5)
    )

    fig.update_layout(
        title=title,
        xaxis_title="Lag (Months)",
        yaxis_title="Partial Autocorrelation",
        template="plotly_white",
        autosize=False,
        width=1200,
        height=850,
        font=dict(size=34),
        margin=dict(l=120, r=60, t=80, b=90)
    )

    fig.update_xaxes(
        tickvals=list(range(0, lags + 1, 4)),
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        showline=True,
        linewidth=1.5,
        linecolor="Gray",
        mirror=True
    )

    fig.update_yaxes(
        range=[-1, 1.1],
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        showline=True,
        linewidth=1.5,
        linecolor="Gray",
        mirror=True
    )

    return fig


def plot_sarimax_results(
    county,
    train_actual,
    train_pred,
    test_actual,
    test_pred
):
    """
    Plot SARIMAX train/test actual and predicted case rates.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_actual.index,
        y=train_actual,
        mode="markers+lines",
        name="Train (True)"
    ))

    fig.add_trace(go.Scatter(
        x=train_pred.index,
        y=train_pred,
        mode="markers+lines",
        name="Train Pred."
    ))

    fig.add_trace(go.Scatter(
        x=test_actual.index,
        y=test_actual,
        mode="markers+lines",
        name="Test (True)"
    ))

    fig.add_trace(go.Scatter(
        x=test_pred.index,
        y=test_pred,
        mode="markers+lines",
        name="Test Pred."
    ))

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Case Rate",
        autosize=False,
        width=1300,
        height=1080,
        font=dict(size=34),
        template="plotly_white",
        xaxis=dict(gridcolor="rgba(0,0,0,0.15)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.15)"),
        legend=dict(
            x=1,
            y=0.85,
            xanchor="right",
            yanchor="bottom",
            bgcolor="white",
            bordercolor="black",
            borderwidth=2
        )
    )

    fig.update_traces(
        line=dict(width=7),
        marker=dict(size=14)
    )

    return fig
