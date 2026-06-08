import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import pacf
import numpy as np

def plot_vf2(county, train_dates, train_actuals, train_predictions,
             test_dates, test_actuals, test_predictions):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_actuals,
        mode="markers+lines",
        name="Train (True)"
    ))

    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_predictions,
        mode="markers+lines",
        name="Train Pred."
    ))

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_actuals,
        mode="markers+lines",
        name="Test (True)"
    ))

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_predictions,
        mode="markers+lines",
        name="Test Pred."
    ))

    fig.update_layout(
        title="",
        xaxis_title="Time",
        yaxis_title="Case Rate",
        template="plotly_white",
        autosize=False,
        width=1300,
        height=1080,
        font=dict(size=34),
        margin=dict(l=120, r=120, t=80, b=90),

    legend=dict(
      x=0.98,
      y=0.98,
      xanchor="right",
      yanchor="top",
      bgcolor="white",
      bordercolor="Gray",
      borderwidth=1.5
  )
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        showline=True,
        linewidth=1.5,
        linecolor="Gray",
        mirror=True,
        range=[
        pd.to_datetime(min(train_dates)) - pd.DateOffset(days=20),
        pd.to_datetime(max(test_dates)) - pd.DateOffset(days=15)
    ]
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        showline=True,
        linewidth=1.5,
        linecolor="Gray",
        mirror=True
    )

    fig.update_traces(
        line=dict(width=6),
        marker=dict(size=11)
    )

    return fig
    
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

def rename_pfi_features(df):
    """
    Rename raw feature names into cleaner labels for PFI plots.
    """
    rename_dict = {
        "cv_mentions_rate": "Cen_Valley_Mentions_Rate",
        "risk_mentions_rate": "Risk_Mentions_Rate",
    }

    df = df.copy()
    df["feature"] = df["feature"].replace(rename_dict)

    return df


def plot_vf2(county, train_dates, train_actuals, train_predictions,
             test_dates, test_actuals, test_predictions):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_actuals,
        mode="markers+lines",
        name="Train (True)"
    ))

    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_predictions,
        mode="markers+lines",
        name="Train Pred."
    ))

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_actuals,
        mode="markers+lines",
        name="Test (True)"
    ))

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_predictions,
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


def plot_mean_pfi2(df, county_name, bar_color="royalblue"):
    """
    Plot average PFI scores across repeated LSTM runs.
    Media-related features are highlighted in red.
    """

    if "county" in df.columns:
        df = df[df["county"] == county_name].copy()

    mean_importance = (
        df.groupby("feature")["pfi"]
        .mean()
        .sort_values(ascending=True)
    )

    features = mean_importance.index.tolist()
    values = mean_importance.values.tolist()

    highlight_labels = [
        "Cen_Valley_Mentions_Rate",
        "Num_Articles",
        "Risk_Mentions_Rate"
    ]

    styled_features = [
        f"<b><span style='color:#aa2e21'>{f}</span></b>"
        if f in highlight_labels else f
        for f in features
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=values,
        y=styled_features,
        orientation="h",
        marker=dict(
            color=bar_color,
            line=dict(color="black", width=2)
        ),
        showlegend=False
    ))

    fig.update_layout(
        xaxis_title="Average Increase in RMSE after Permutation",
        yaxis_title="Feature",
        template="plotly_white",
        autosize=False,
        width=1200,
        height=1080,
        font=dict(size=34),
        margin=dict(l=500)
    )

    fig.update_xaxes(
        range=[-0.08, 0.14],
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    )

    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True
    )

    return fig


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


