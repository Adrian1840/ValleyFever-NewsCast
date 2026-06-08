import pandas as pd
import plotly.graph_objects as go


def plot_combined_vf_styled(county, df):
    df = df.sort_values("Year-Month")

    fig = go.Figure()

    bar_color = "#7499d4"
    line_color = "#c22f53"

    fig.add_trace(go.Bar(
        x=df["Year-Month"],
        y=df["VFRate"],
        name="VF Case Rates",
        marker_color=bar_color,
        opacity=0.9,
        yaxis="y"
    ))

    fig.add_trace(go.Scatter(
        x=df["Year-Month"],
        y=df["Num_Articles"],
        name="News Articles",
        mode="markers+lines",
        line=dict(color=line_color, width=6),
        marker=dict(size=11, color=line_color),
        yaxis="y2"
    ))

    fig.update_layout(
        title="",
        xaxis_title="Time",
        template="plotly_white",
        autosize=False,
        width=1920,
        height=1080,
        font=dict(size=34),
        margin=dict(l=120, r=120, t=80, b=90),

        yaxis=dict(
            title="Case Rate",
            showgrid=True,
            gridwidth=1,
            gridcolor="Gray",
            griddash="dash",
            showline=True,
            linewidth=1.5,
            linecolor="Gray",
            mirror=True
        ),

        yaxis2=dict(
            title="Number of Articles",
            overlaying="y",
            side="right",
            showgrid=False,
            showline=True,
            linewidth=1.5,
            linecolor="Gray",
            mirror=True
        ),

        legend=dict(
            x=0.04,
            y=0.97,
            xanchor="left",
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
        pd.to_datetime(df["Year-Month"].min()) - pd.DateOffset(days=20),
        pd.to_datetime(df["Year-Month"].max()) + pd.DateOffset(days=20)
    ]
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
