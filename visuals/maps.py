import pandas as pd
import plotly.express as px


# =========================================================
# STATE NAME NORMALIZATION (CRITICAL FOR GEOJSON MATCHING)
# =========================================================

STATE_NAME_MAP = {
    "NCT of Delhi": "Delhi",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Odisha": "Orissa",
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "Dadra & Nagar Haveli and Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu"
}


def normalize_state_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["state"] = df["state"].replace(STATE_NAME_MAP)
    return df


# =========================================================
# MAP 1: STATE TRANSACTION DENSITY
# =========================================================

def state_transaction_map(state_df: pd.DataFrame):
    """
    State-wise Aadhaar transaction density map.
    Answers: Where is operational load concentrated?
    """

    df = normalize_state_names(state_df)
    df["count"] = df.groupby("state")["count"].transform("sum")
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/rajkumarkanagasingam/india-geojson/master/india_states.geojson",
        featureidkey="properties.ST_NM",
        locations="state",
        color="count",
        color_continuous_scale="Blues",
        title="State-wise Aadhaar Transaction Density"
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


# =========================================================
# MAP 2: STATE UPDATE PRESSURE (MAINTENANCE HEATMAP)
# =========================================================

def state_update_pressure_map(ratio_df: pd.DataFrame):
    """
    State-wise update-to-enrolment pressure map.
    Answers: Which states are maintenance-heavy?
    """

    df = normalize_state_names(ratio_df)

    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/rajkumarkanagasingam/india-geojson/master/india_states.geojson",
        featureidkey="properties.ST_NM",
        locations="state",
        color="update_to_enrolment_ratio",
        color_continuous_scale="Reds",
        title="State-wise Aadhaar Maintenance Pressure"
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig
