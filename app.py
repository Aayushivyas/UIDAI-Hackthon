import streamlit as st
import pandas as pd

# =========================================================
# PROCESSING (KPIs ONLY)
# =========================================================
from processing.loaders import build_transaction_fact
from processing.metrics import executive_kpis
from analytics.model import (
    _parse_date,
    _age_cols
)
# =========================================================
# VISUALS
# =========================================================
from visuals.kpis import render_kpis
from visuals.charts import (
    state_workload_chart,
    pareto_chart,
    spike_timeline_chart,
    instability_severity_chart,
    monthly_anomalies_chart,
    cluster_pca_chart,
    cluster_profile_chart
)
from visuals.maps import state_transaction_map
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Aadhaar Operations Dashboard",
    layout="wide"
)

st.title("Aadhaar Enrollment & Operations Dashboard")

# =========================================================
# DATA INGESTION
# =========================================================
DATA_DIR = r"D:\Hackathon\data"

enrollment_df = pd.read_csv(
    rf"{DATA_DIR}\api_data_aadhar_enrolment_full.csv"
)
demographic_df = pd.read_csv(
    rf"{DATA_DIR}\api_data_aadhar_demographic_full.csv"
)
biometric_df = pd.read_csv(
    rf"{DATA_DIR}\api_data_aadhar_biometric_full.csv"
)

# Basic validation
if (
    enrollment_df.empty
    or demographic_df.empty
    or biometric_df.empty
):
    st.error("One or more required datasets are empty.")
    st.stop()

# Date parsing
enrollment_df = _parse_date(enrollment_df)
# enrollment_df = _age_cols(enrollment_df)
# =========================================================
# KPI COMPUTATION (ALL DATASETS)
# =========================================================
fact_df = build_transaction_fact(
    enrollment_df,
    demographic_df,
    biometric_df
)

kpis = executive_kpis(fact_df)
render_kpis(kpis)

st.divider()

# =========================================================
# FILTERS (ENROLLMENT ONLY)
# =========================================================
st.sidebar.header("Filters")

state_filter = st.sidebar.multiselect(
    "State",
    sorted(enrollment_df["state"].unique())
)

filtered_enrollment_df = (
    enrollment_df[enrollment_df["state"].isin(state_filter)]
    if state_filter else enrollment_df
)

# =========================================================
# OPERATIONAL OVERVIEW (ENROLLMENT ONLY)
# =========================================================
st.subheader("Operational Overview (Enrollment Dataset Only)")
# st.write("Enrollment columns:", list(filtered_enrollment_df.columns))
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        state_workload_chart(filtered_enrollment_df),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        pareto_chart(filtered_enrollment_df),
        use_container_width=True
    )

st.divider()

# =========================================================
# ADVANCED ANALYTICS (ENROLLMENT ONLY)
# =========================================================
st.subheader("Advanced Analytics")

show_advanced = st.checkbox(
    "Show Advanced Analytics (Monthly Aggregation)",
    value=False
)

if show_advanced:

    # -----------------------------------------------------
    # SPIKE DETECTION
    # -----------------------------------------------------
    st.markdown("### 📈 Monthly Enrollment Spikes")

    st.plotly_chart(
        spike_timeline_chart(filtered_enrollment_df),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # INSTABILITY ANALYSIS
    # -----------------------------------------------------
    st.markdown("### ⚠️ District Instability vs Severity")

    st.plotly_chart(
        instability_severity_chart(filtered_enrollment_df),
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # MONTHLY ANOMALIES
    # -----------------------------------------------------
    st.markdown("### 🚨 Top Monthly Anomalies")

    st.plotly_chart(
        monthly_anomalies_chart(filtered_enrollment_df),
        use_container_width=True
    )

    st.divider()

    

    