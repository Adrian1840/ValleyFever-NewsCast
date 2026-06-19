import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import pacf
from plotly.subplots import make_subplots
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


def plot_vf_combined(
        county,
        train_dates,
        train_actuals,
        train_predictions,
        test_dates,
        test_actuals,
        test_predictions,
        sarimax_test_predictions=None):

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.08
    )

    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_actuals,
        mode="markers+lines",
        name="Train (True)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_predictions,
        mode="markers+lines",
        name="LSTM Train Pred."
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_actuals,
        mode="markers+lines",
        name="Test (True)"
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_predictions,
        mode="markers+lines",
        name="LSTM Test Pred."
    ), row=1, col=2)

    if sarimax_test_predictions is not None:
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=sarimax_test_predictions,
            mode="markers+lines",
            name="SARIMAX Test Pred."
        ), row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        autosize=False,
        width=1920,
        height=1080,
        font=dict(size=34),
        margin=dict(l=120, r=120, t=80, b=90),

        legend=dict(
            x=0.44,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1.5,
            font=dict(size=28)
        )
    )

    fig.update_xaxes(
        title_text="Time",
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
        title_text="Case Rate",
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

     # Testing split x-axis
    fig.update_xaxes(
        tickformat="%b %Y",
        nticks=2,
        row=1,
        col=2
    )

    return fig
    
    

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
        title="",
        xaxis_title="Time",
        yaxis_title="Case Rate",
        template="plotly_white",
        autosize=False,
        width=1920,
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
            pd.to_datetime(train_actual.index.min()) - pd.DateOffset(days=20),
            pd.to_datetime(test_actual.index.max()) + pd.DateOffset(days=20)
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


