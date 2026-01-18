import pandas as pd

# =========================================================
# EXECUTIVE KPIs (SINGLE SOURCE OF TRUTH)
# =========================================================

def executive_kpis(df: pd.DataFrame) -> dict:
    """
    High-level KPIs for executive dashboard.
    """

    total_transactions = df["count"].sum()

    enrollments = df.loc[
        df["transaction_type"] == "enrolment", "count"
    ].sum()

    updates = df.loc[
        df["transaction_type"] != "enrolment", "count"
    ].sum()

    update_pct = (
        (updates / total_transactions) * 100
        if total_transactions > 0 else 0
    )

    # State concentration
    state_totals = (
        df.groupby("state")["count"]
        .sum()
        .sort_values(ascending=False)
    )

    top_state_share = (
        (state_totals.iloc[0] / total_transactions) * 100
        if total_transactions > 0 and len(state_totals) > 0
        else 0
    )

    # HHI-style concentration index
    concentration_index = (
        ((state_totals / total_transactions) ** 2).sum()
        if total_transactions > 0 else 0
    )
    
    return {
        "total_transactions": round(float(total_transactions), 2),
        "enrollments": int(enrollments),
        "updates": int(updates),
        "update_pct": round(update_pct, 2),
        "top_state_share": round(top_state_share, 2),
        "concentration_index": round(concentration_index, 4)
    }


# =========================================================
# SUPPORTING METRICS (USED BY CHARTS)
# =========================================================

def state_workload(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("state", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )


def update_enrolment_ratio_by_state(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby(["state", "transaction_type"])["count"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    pivot["update_to_enrolment_ratio"] = (
        pivot.drop(columns=["state", "enrolment"], errors="ignore").sum(axis=1)
        / pivot["enrolment"].replace(0, pd.NA)
    )

    return pivot.sort_values(
        "update_to_enrolment_ratio",
        ascending=False
    )


def age_group_pressure(df: pd.DataFrame) -> pd.DataFrame:
    enrol = (
        df[df["transaction_type"] == "enrolment"]
        .groupby("age_group")["count"]
        .sum()
    )

    updates = (
        df[df["transaction_type"] != "enrolment"]
        .groupby("age_group")["count"]
        .sum()
    )

    pressure = (updates / enrol).reset_index()
    pressure.columns = ["age_group", "update_pressure"]

    return pressure.sort_values(
        "update_pressure",
        ascending=False
    )


def transaction_mix(df: pd.DataFrame) -> pd.DataFrame:
    mix = (
        df.groupby("transaction_type")["count"]
        .sum()
        .reset_index()
    )

    total = mix["count"].sum()
    mix["percentage"] = (mix["count"] / total) * 100

    return mix.sort_values("count", ascending=False)


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.assign(month=df["date"].dt.to_period("M"))
        .groupby(["month", "transaction_type"])["count"]
        .sum()
        .reset_index()
    )

    monthly["month"] = monthly["month"].astype(str)
    return monthly


def district_load(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["state", "district"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
    )
