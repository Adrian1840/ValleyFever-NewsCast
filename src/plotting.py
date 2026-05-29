
import plotly.graph_objects as go


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
