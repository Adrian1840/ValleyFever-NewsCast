
import plotly.graph_objects as go


def plot_vf2(county, train_dates, train_actuals, train_predictions,
             test_dates, test_actuals, test_predictions):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_dates, y=train_actuals,
        mode="markers+lines", name="Train (True)"
    ))

    fig.add_trace(go.Scatter(
        x=train_dates, y=train_predictions,
        mode="markers+lines", name="Train Pred."
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=test_actuals,
        mode="markers+lines", name="Test (True)"
    ))

    fig.add_trace(go.Scatter(
        x=test_dates, y=test_predictions,
        mode="markers+lines", name="Test Pred."
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
