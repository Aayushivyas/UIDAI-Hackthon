import streamlit as st

# -------------------------------------------------
# Compact number formatter (K / M / B)
# -------------------------------------------------
def _format_compact(value):
    if value is None:
        return "—"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    abs_val = abs(value)

    if abs_val >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{int(value):,}"

# -------------------------------------------------
# Executive KPI ribbon
# -------------------------------------------------
def render_kpis(kpis: dict):

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Total Transactions",
        _format_compact(kpis.get("total_transactions"))
    )

    col2.metric(
        "Enrollments",
        _format_compact(kpis.get("enrollments"))
    )

    col3.metric(
        "Updates",
        _format_compact(kpis.get("updates"))
    )

    col4.metric(
        "Update %",
        f"{kpis['update_pct']:.1f}%" if kpis.get("update_pct") is not None else "—"
    )

    col5.metric(
        "Top State Share",
        f"{kpis['top_state_share']:.1f}%" if kpis.get("top_state_share") is not None else "—"
    )

    col6.metric(
        "Concentration Index",
        f"{kpis['concentration_index']:.4f}"
        if kpis.get("concentration_index") is not None else "—"
    )
