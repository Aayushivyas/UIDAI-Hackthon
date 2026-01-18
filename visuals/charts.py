import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analytics.model import (
    monthly_aggregate,
    _age_cols,
    add_spike_metrics,
    top_anomalies_by_month,
    instability_score_with_severity,
    kmeans_cluster_districts,
    plot_cluster_profile,
    plot_clusters_pca
)

# =========================================================
# STATE WORKLOAD (ENROLLMENT ONLY)
# =========================================================
def state_workload_chart(enrollment_df: pd.DataFrame):
    df = enrollment_df.copy()

    df["total"] = (
        df["age_0_5"]
        + df["age_5_17"]
        + df["age_18_greater"]
    )

    state_df = (
        df.groupby("state", as_index=False)["total"]
        .sum()
        .sort_values("total", ascending=False)
    )

    return px.bar(
        state_df,
        x="state",
        y="total",
        title="State-wise Enrollment Volume"
    )

# =========================================================
# PARETO (TOP STATE CONTRIBUTION)
# =========================================================
def pareto_chart(enrollment_df: pd.DataFrame):
    df = enrollment_df.copy()

    df["total"] = (
        df["age_0_5"]
        + df["age_5_17"]
        + df["age_18_greater"]
    )

    state_df = (
        df.groupby("state", as_index=False)["total"]
        .sum()
        .sort_values("total", ascending=False)
    )

    state_df["cum_pct"] = (
        state_df["total"].cumsum()
        / state_df["total"].sum()
        * 100
    )

    fig = go.Figure()

    # Bar: Enrollment volume
    fig.add_bar(
        x=state_df["state"],
        y=state_df["total"],
        name="Enrollment Volume",
        yaxis="y1"
    )

    # Line: Cumulative %
    fig.add_scatter(
        x=state_df["state"],
        y=state_df["cum_pct"],
        name="Cumulative %",
        yaxis="y2",
        mode="lines+markers"
    )

    fig.update_layout(
        title="Pareto Analysis: Enrollment Volume by State",
        xaxis=dict(title="State"),
        yaxis=dict(
            title="Enrollment Volume",
            showgrid=False
        ),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100],
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

# =========================================================
# SPIKE TIMELINE
# =========================================================

def spike_timeline_chart(enrollment_df: pd.DataFrame):
    val_cols = _age_cols(enrollment_df)
    monthly_df = monthly_aggregate(enrollment_df, 'district', val_cols)
    spike_df = add_spike_metrics(monthly_df, group_keys=["state", "district"], value_col="total")
    
    # Create base line chart
    fig = px.line(
        spike_df,
        x="month",
        y="total",
        color="district",
        title="Monthly Enrollment Spikes",
        labels={"total": "Total Enrollment", "month": "Month"}
    )
    
    # Add markers for spikes
    spike_points = spike_df[spike_df["is_spike"] == True]
    fig.add_scatter(
        x=spike_points["month"],
        y=spike_points["total"],
        mode="markers",
        marker=dict(color="red", size=10, symbol="diamond"),
        name="Spike Detected",
        showlegend=True
    )
    
    # Optional: Adjust layout
    fig.update_layout(
        hovermode="x unified",
        legend_title_text="District / Spike"
    )
    
    return fig

# =========================================================
# INSTABILITY + SEVERITY
# =========================================================
def instability_severity_chart(enrollment_df: pd.DataFrame):
    val_cols = _age_cols(enrollment_df)
    monthly_df = monthly_aggregate(enrollment_df, 'district', val_cols)
    spike_df = add_spike_metrics(monthly_df, group_keys=["state", "district"], value_col="total")
    df = instability_score_with_severity(
        monthly_df,
        spike_df,
        level="district",
        severity_agg="sum",
        severity_stat="p95",
        peak_mode="ratio"
    )
    # st.write("Instability DataFrame:", list(df.columns))
    return px.scatter(
        df,
        x="instability_score",
        y="severity_norm",
        size="max_total",
        color="spike_freq",
        hover_name="district",
        title="District Instability vs Severity"
    )


# =========================================================
# MONTHLY ANOMALIES
# =========================================================
def monthly_anomalies_chart(enrollment_df: pd.DataFrame):
    val_cols = _age_cols(enrollment_df)
    monthly_df = monthly_aggregate(enrollment_df, 'district', val_cols)
    spike_df = add_spike_metrics(monthly_df, group_keys=["state", "district"], value_col="total")
    anomalies = top_anomalies_by_month(spike_df, topn=5)

    return px.bar(
        anomalies,
        x="month",
        y="severity",
        color="district",
        title="Top Monthly Enrollment Anomalies"
    )


# # =========================================================
# # CLUSTERING
# # =========================================================
@st.cache_data(show_spinner=False)
def _cluster_cached(enrollment_df: pd.DataFrame, k: int):
    val_cols = _age_cols(enrollment_df)
    monthly_df = monthly_aggregate(enrollment_df, 'district', val_cols)
    spike_df = add_spike_metrics(monthly_df, group_keys=["state", "district"], value_col="total")
    instability_df = instability_score_with_severity(
        monthly_df,
        spike_df,
        level="district",
        severity_agg="sum",
        severity_stat="p95",
        peak_mode="ratio"
    )
    return kmeans_cluster_districts(instability_df,feature_cols=["cov_norm", "spike_norm", "peak_norm", "severity_norm"], k=k)

def cluster_pca_chart(enrollment_df: pd.DataFrame, k: int = 4):
    features = ["avg_cov_norm","avg_spike_norm", "avg_peak_norm", "avg_severity_norm"]
    cluster_df, pca_df = _cluster_cached(enrollment_df, k)
    # st.write("PCA DataFrame Columns:", list(pca_df.columns))
    return plot_clusters_pca(pca_df,feature_cols=features, title="Enrollment Instability — Pincode Clusters",
    hover_cols=["state", "district", "pincode", "instability_score", "cluster"])


def cluster_profile_chart(enrollment_df: pd.DataFrame, k: int = 4):
    cluster_df, _ = _cluster_cached(enrollment_df, k)
    features = ["cov_norm", "spike_norm", "peak_norm", "severity_norm"]
    return plot_cluster_profile(cluster_df, feature_cols=features, title="Enrollment Instability — Cluster Profiles (Pincode)")
