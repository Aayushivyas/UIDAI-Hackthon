# %%
#Preliminary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Optional, Dict
# %% [markdown]
# ### **AADHAR ENROLMENT DATASET**

# %% [markdown]
# This dataset provides aggregated information on **Aadhaar enrolments** across various demographic and geographic levels. It includes variables such as the date of enrollment, state, district, PIN code, and age-wise categories (0–5 years, 5–17 years, and 18 years and above). The dataset captures both temporal and spatial patterns of enrolment activity, enabling detailed descriptive, comparative, and trend analysis.


# %% [markdown]
# ### **Aadhaar Demographic Update dataset**

# %% [markdown]
# This dataset captures aggregated information related to **updates made to residents’ demographic data linked to Aadhaar**, such as name, address, date of birth, gender, and mobile number. It provides insights into the frequency and distribution of demographic changes across different time periods and geographic levels (state, district, and PIN code).


# %% [markdown]
# ### **Aadhaar Biometric Update dataset**

# %% [markdown]
# This dataset contains aggregated information on biometric updates (modalities such as fingerprints, iris, and face). It reflects the periodic revalidation or correction of biometric details, especially for children transitioning into adulthood.

# %% [markdown]
# ### **Parsing Date**

# %%
def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()  # month start
    return df


# %% [markdown]
# ### **Age column to numeric**

# %%
#age column to numeric
def _age_cols(df: pd.DataFrame):
    base = {"date", "month", "state", "district", "pincode"}
    cols = [c for c in df.columns if c not in base]
    # keep only numeric-like columns
    num_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].astype(str).str.match(r"^-?\d+(\.\d+)?$").mean() > 0.8:
            num_cols.append(c)
    # force numeric
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return num_cols


# %% [markdown]
# ### **Monthly Aggregate at Pin/District**

# %%
def monthly_aggregate(df: pd.DataFrame, level: str, value_cols: list[str]) -> pd.DataFrame:
    """
    level: "pin" or "district"
    returns monthly sums by level.
    """
    if level == "pin":
        keys = ["month", "state", "district", "pincode"] #group by month + state + district + pincode
    elif level == "district":
        keys = ["month", "state", "district"]            #group by month + state + district
    else:
        raise ValueError("level must be 'pin' or 'district'")

    out = (
        df.groupby(keys, as_index=False)[value_cols]
          .sum()
          .sort_values(keys)
          .reset_index(drop=True)
    )
    # total across age columns (handy for spikes/forecast)
    out["total"] = out[value_cols].sum(axis=1)
    return out


# ----------------------------

# %% [markdown]
# ### **System wide Aadhar activity by Age Group**

# %%


# %% [markdown]
# ### **Spike detection/Severity Scoring**

# %% [markdown]
# The severity score is constructed as a multiplicative composite of three interpretable components: 
# 
# * relative magnitude (absolute month-over-month percent change), 
# * statistical unusualness (a robust z-score of the month-to-month change using median and MAD), and 
# * impact/scale (a log-transformed current-month volume). 
# 
# This design reflects a standard “risk = likelihood × impact” philosophy: the robust z-score captures how unlikely the change is relative to the location’s own historical behavior (reducing sensitivity to outliers and non-normality), the percent change captures how large the shift is in relative terms, and the log-volume term ensures the score reflects real-world operational impact without letting very large regions dominate purely by size.
# 
# Severity score can be as a prioritization and response tool. The score helps identify which events demand immediate attention because they combine large change, rarity, and high operational impact. High-severity spikes can trigger actions such as surge staffing, temporary capacity expansion, targeted audits, or advance planning for expected drives. Aggregated over time and locations, severity scores also reveal districts that experience intense stress during spike months, informing where contingency plans, buffers, and preventive interventions will deliver the greatest benefit.

# %%
def add_spike_metrics(monthly_df: pd.DataFrame, group_keys: list[str], value_col: str = "total") -> pd.DataFrame:
    """
    Adds MoM absolute/percent changes + robust z-score on change + severity score.
    severity = (abs(pct_change) capped) * robust_z_change * log1p(current_value)
    """
    df = monthly_df.copy()
    df = df.sort_values(group_keys + ["month"])

    # MoM changes per group
    #Create a groupby object (one group = one PIN time series
    #g represents multiple separate time series:
    #one for each unique (state, district, pincode) combination.
    g = df.groupby(group_keys, sort=False)

    #For each group (PIN let's say)
    #-moves the values down by 1 row
    #-so current row can reference its previous month’s value
    df["prev"] = g[value_col].shift(1)                #last month’s total
    df["abs_change"] = df[value_col] - df["prev"]     #current_total − prev_total
    #If a PIN has its first month in the data: prev becomes NaN because there’s no earlier row.

    # percent change (safe)
    df["pct_change"] = np.where(df["prev"].fillna(0) == 0, np.nan, df["abs_change"] / df["prev"])

    # robust z-score of abs_change within each group (using MAD)
    #It takes a series of changes within one group (PIN) and scores how unusual each change is compared to that PIN’s own history.
    #What it is computing -- A) med: median of the series  B) mad: median absolute deviation from the median
    #robust z = (x − median) / (1.4826 * MAD)
    def _robust_z(s: pd.Series) -> pd.Series:
        med = np.nanmedian(s)
        mad = np.nanmedian(np.abs(s - med))
        if mad == 0 or np.isnan(mad):
            return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        return (s - med) / (1.4826 * mad) #Evaluating the z_score while 1.4826 * MAD = Standard Deviation for normal distribution

    df["z_change"] = g["abs_change"].transform(_robust_z).abs() #Passing absolute change series through _robust_z
    #z_change ~ 0 --> Normal; z_change >=3 --> Significant anomaly


    # cap pct_change to avoid infinite blow-ups from tiny denominators
    pct = df["pct_change"].abs().clip(lower=0, upper=20)  # bounded percent-change magnitude
    
    
    #Severity score ranks regions by how unusually large and operationally impactful their recent activity changes are. 
   
    df["severity"] = (pct.fillna(0) + 0.1) * (df["z_change"].fillna(0) + 0.1) * np.log1p(df[value_col])
    #pct --> How big was the month-to-month change in percentage?
    #z_change --> How unusual was this change for this PIN historically or how surprising is this change for this area?
    #log(1+x) --> How big is the real-world impact of this anomaly without letting large regions overpower the analysis?
    # If either pct or z_change is 0, the product would become 0 and you’d lose information.+ 0.1 makes severity still non-zero for mild anomalies and prevents hard zeros.

    # flag spikes (TRUE if)
    #Condition 1: Top 1% by severity globally --> severity >= 99th percentile, OR
    #Condition 2: z_change >= 3 (very unusual for that PIN) AND pct >= 1 (≥ 100% change)
    
    df["is_spike"] = (df["severity"] >= df["severity"].quantile(0.99)) | ((df["z_change"] >= 3) & (pct >= 1))
    return df


# %% [markdown]
# ### **Top anomalies by month**

# %% [markdown]
# Anomalies typically represent unusual surges, drops, or disruptions that cannot be explained by normal variability. In this analysis, anomaly detection is applied at a granular month-level to measure how different a given month’s activity is compared to that location’s own historical behavior, rather than against a global average.
# 
# Methodologically, the approach combines month-to-month change, relative percentage change, and a robust z-score based on the median absolute deviation (MAD), which is well established in the literature as a stable alternative to standard deviation when data contain outliers or non-normal distributions. This allows the model to detect true operational shocks while remaining resistant to noise and scale effects across districts of different sizes. Each detected anomaly is further quantified using a severity score, which integrates the magnitude of change, its statistical rarity, and the underlying activity volume, ensuring that both unusualness and real-world impact are captured.
# 
# The utility of anomaly detection lies in its ability to produce actionable, time-specific signals—highlighting when and where abnormal events occur, such as special enrolment drives, system outages, or backlog clearances. These anomaly signals form the evidentiary layer of the analysis and are subsequently aggregated to inform higher-level risk metrics, such as spike frequency and severity in the Instability Score. In this way, anomaly detection serves as the event-level diagnostic foundation, complementing district-level instability analysis and enabling both immediate operational response and longer-term planning.

# %%
def top_anomalies_by_month(spike_df: pd.DataFrame, topn: int = 20) -> pd.DataFrame:
    """
    Returns top N spike anomalies for each month ranked by severity (descending).
    Works for both PIN-level spikes (has pincode) and district-level spikes (no pincode).
    Only includes rows where is_spike == True.
    """
    df = spike_df.copy()

    # keep valid months and only spikes
    df = df[df["month"].notna()].copy()
    df = df[df["is_spike"] == True].copy()

    if df.empty:
        # return empty dataframe with whatever columns would have existed
        return df

    # rank within month by severity (highest = rank 1)
    df["rank_in_month"] = df.groupby("month")["severity"].rank(method="first", ascending=False)

    out = (
        df[df["rank_in_month"] <= topn]
        .sort_values(["month", "rank_in_month"], ascending=[True, True])
        .reset_index(drop=True)
    )

    # Choose identifier columns based on availability
    id_cols = ["month", "state", "district"]
    if "pincode" in out.columns:
        id_cols.append("pincode")

    # Common metric columns (keep only those that exist)
    metric_cols = [
        "total", "prev", "abs_change", "pct_change", "z_change",
        "severity", "is_spike", "rank_in_month"
    ]
    keep_cols = [c for c in (id_cols + metric_cols) if c in out.columns]

    return out[keep_cols]




# %% [markdown]
# ### **Instability Score**

# %% [markdown]
# Instability Score is a composite indicator that summarizes how operationally unpredictable a district’s Aadhaar activity is over time. Rather than relying on raw monthly volumes, it integrates four complementary signals: 
# 
# * volatility (how much monthly activity fluctuates relative to its average), 
# * spike frequency (how often abnormal surges occur), 
# * peak stress (how extreme the highest-load month is compared to normal), and 
# * spike intensity (how severe those abnormal surges are when they occur). 
# 
# Together, these dimensions capture both chronic instability and event-driven risk, an approach widely used in operations and risk management literature where composite indices are favored for their robustness and interpretability over single metrics.
# 
# For a decision maker, the Instability Score functions as a prioritization and planning tool. A higher score indicates districts or pins that are harder to manage operationally and more prone to service disruptions without proactive intervention.
# 
# The score translates complex time-series behavior into a single, actionable measure that supports informed resource allocation, risk mitigation, and policy decisions.

# %%
"""
    level="district" -> one row per (state, district)
    level="pin"      -> one row per (state, district, pincode)

    Returns one row per (state, district/pin) with:
      - cov (weight = 0.30): {Std/Mean}, 
      - spike_freq (weight = 0.25): {is_spike = TRUE / no of months}, 
      - peak_factor (weight = 0.20): {Max Total / Mean}
      - severity metric (weight = 0.25): aggregated severity of spikes in that district
      - normalized components
      - final instability_score (0-100)

    Inputs:
      monthly_df: output of monthly_aggregate(..., level="district") OR similar
      spike_df: output of add_spike_metrics() at district or pin level

     severity_agg:
      - sum: total severity across spiky PINs in a district-month (captures distributed stress)
      - max: worst spike in that district-month (captures extreme local pocket)
      - mean: average severity across spiky PINs (captures typical spike intensity), how intense is a typical spike pocket?
      
    severity_stat = mean: 
        What it emphasizes
        - Chronic, everyday stress
        - Repeated moderate problems
        - Long-term operational load
        What it ignores
        - Rare but very dangerous months
        - Extreme one-off stress events
    severity_stat = p95:
        What it emphasizes
        - Risk of extreme stress events
        - Preparedness for worst-case scenarios
        - Planning for rare but high-impact spikes
        What it ignores
        - Everyday operational load
        - Chronic moderate problems
        - Normal months
    """


import numpy as np
import pandas as pd

def _robust_minmax(s: pd.Series, low_q=0.05, high_q=0.95) -> pd.Series:
    s = s.astype(float)
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s.clip(lo, hi) - lo) / (hi - lo)

def instability_score_with_severity(
    monthly_df: pd.DataFrame,
    spike_df: pd.DataFrame,
    level: str = "district",            # "district" or "pin"
    value_col: str = "total",
    min_months: int = 4,
    weights: Optional[Dict] = None,
    severity_agg: str = "sum",          # "sum" or "max" or "mean"
    severity_stat: str = "p95",         # "mean" or "p95"
    peak_mode: str = "ratio"            # "ratio" or "log_ratio"
) -> pd.DataFrame:
   
    # ---- Defaults
    if weights is None:
        # Must sum to 1.0 for clean 0–100 interpretation
        weights = {"cov": 0.30, "spike_freq": 0.25, "peak": 0.20, "severity": 0.25}

    # ---- Validate mode params
    if level not in {"district", "pin"}:
        raise ValueError("level must be 'district' or 'pin'")
    if severity_agg not in {"sum", "max", "mean"}:
        raise ValueError("severity_agg must be one of: 'sum', 'max', 'mean'")
    if severity_stat not in {"mean", "p95"}:
        raise ValueError("severity_stat must be one of: 'mean', 'p95'")
    if peak_mode not in {"ratio", "log_ratio"}:
        raise ValueError("peak_mode must be 'ratio' or 'log_ratio'")

    # ---- Group keys
    keys = ["state", "district"] if level == "district" else ["state", "district", "pincode"]

    # ---- Validate required columns
    required_monthly = set(keys + ["month", value_col])
    missing_monthly = required_monthly - set(monthly_df.columns)
    if missing_monthly:
        raise ValueError(f"monthly_df missing required columns: {missing_monthly}")

    required_spike = set(keys + ["month", "is_spike", "severity"])
    missing_spike = required_spike - set(spike_df.columns)
    if missing_spike:
        raise ValueError(f"spike_df missing required columns: {missing_spike}")

    # ---- Sort monthly data
    df = monthly_df.copy().sort_values(keys + ["month"])

    # 1) Base metrics per group
    base = (
        df.groupby(keys, as_index=False)
          .agg(
              months=("month", "nunique"),
              mean=(value_col, "mean"),
              std=(value_col, "std"),
              max_total=(value_col, "max"),
          )
    )

    base["cov"] = np.where(base["mean"] == 0, np.nan, base["std"] / base["mean"])
    base["peak_factor"] = np.where(base["mean"] == 0, np.nan, base["max_total"] / base["mean"])
    if peak_mode == "log_ratio":
        base["peak_factor"] = np.log1p(base["peak_factor"])

    # Ensure enough history
    base = base[base["months"] >= min_months].copy()

    # 2) Spike frequency (align spikes to months present in monthly_df)
    spikes_only = spike_df[spike_df["is_spike"] == True].copy()

    # Keep only months that exist in monthly_df for the same group (prevents time-range mismatch)
    valid_months = df[keys + ["month"]].drop_duplicates()
    spikes_only = spikes_only.merge(valid_months, on=keys + ["month"], how="inner")

    spike_months = (
        spikes_only.drop_duplicates(subset=keys + ["month"])
        .groupby(keys, as_index=False)
        .agg(spike_months=("month", "nunique"))
    )

    out = base.merge(spike_months, on=keys, how="left")
    out["spike_months"] = out["spike_months"].fillna(0).astype(int)
    out["spike_freq"] = np.where(out["months"] == 0, 0.0, out["spike_months"] / out["months"])

    # 3) Severity metric
    # Aggregate severity at group-month level
    if severity_agg == "sum":
        group_month_sev = spikes_only.groupby(keys + ["month"], as_index=False)["severity"].sum()
    elif severity_agg == "max":
        group_month_sev = spikes_only.groupby(keys + ["month"], as_index=False)["severity"].max()
    else:  # mean
        group_month_sev = spikes_only.groupby(keys + ["month"], as_index=False)["severity"].mean()

    group_month_sev = group_month_sev.rename(columns={"severity": "group_month_severity"})

    # Summarize across months per group
    if severity_stat == "mean":
        sev = (
            group_month_sev.groupby(keys, as_index=False)["group_month_severity"]
            .mean()
            .rename(columns={"group_month_severity": "severity_metric"})
        )
    else:  # p95
        sev = (
            group_month_sev.groupby(keys, as_index=False)["group_month_severity"]
            .quantile(0.95)
            .rename(columns={"group_month_severity": "severity_metric"})
        )

    out = out.merge(sev, on=keys, how="left")
    out["severity_metric"] = out["severity_metric"].fillna(0.0)

    # 4) Normalize components
    out["cov_norm"] = _robust_minmax(out["cov"].fillna(0))
    out["spike_norm"] = _robust_minmax(out["spike_freq"].fillna(0))
    out["peak_norm"] = _robust_minmax(out["peak_factor"].fillna(0))
    out["severity_norm"] = _robust_minmax(out["severity_metric"].fillna(0))

    # 5) Final score (0–100)
    out["instability_score"] = 100.0 * (
        weights["cov"] * out["cov_norm"] +
        weights["spike_freq"] * out["spike_norm"] +
        weights["peak"] * out["peak_norm"] +
        weights["severity"] * out["severity_norm"]
    )

    out = out.sort_values("instability_score", ascending=False).reset_index(drop=True)

    cols = (
        keys + [
            "months", "mean", "std", "cov",
            "max_total", "peak_factor",
            "spike_months", "spike_freq",
            "severity_metric",
            "cov_norm", "spike_norm", "peak_norm", "severity_norm",
            "instability_score"
        ]
    )
    return out[cols]

    

# %%
#Instability score at district level



# %% [markdown]
# ### **Top X contribution in instability**

# %% [markdown]
# Top-X contribution analysis is a concentration metric used to quantify how much of a system’s total instability is driven by a small subset of units (e.g., districts or PIN codes). Rather than only identifying the most unstable entities, it measures the cumulative share of instability accounted for by the top X% of units, providing insight into the distributional structure of risk across the system.
# 
# The primary objective of Top-X contribution analysis is to answer the question:
# 
# *“Is instability widespread across the system, or is it concentrated in a small number of hotspots?”*
# 
# By expressing instability in cumulative terms (e.g., “Top 10% of districts contribute 55% of total instability”), this metric enables:
# 
# * Prioritization of interventions, by identifying whether targeting a small subset of districts can significantly reduce system-wide risk.
# * Efficient resource allocation, avoiding uniform responses when problems are structurally concentrated.
# * Strategic planning, by distinguishing between systemic instability and localized operational stress.
# 
# This makes Top-X contribution especially valuable for decision makers who must choose where limited monitoring, staffing, or corrective capacity should be deployed.

# %%
import numpy as np
import pandas as pd

def top_x_contribution(df: pd.DataFrame, value_col: str = "instability_score", x: float = 0.10) -> dict:
    if not (0 < x <= 1):
        raise ValueError("x must be between 0 and 1")

    s = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).clip(lower=0)

    if s.sum() == 0:
        return {"top_share_units_pct": 0.0, "contribution_share_pct": 0.0, "n_top_units": 0}

    sorted_vals = s.sort_values(ascending=False)
    n = len(sorted_vals)
    n_top = max(1, int(np.ceil(x * n)))

    top_sum = sorted_vals.iloc[:n_top].sum()
    total_sum = sorted_vals.sum()

    return {
        "top_share_units_pct": round(100 * n_top / n, 2),
        "contribution_share_pct": round(100 * top_sum / total_sum, 2),
        "n_top_units": n_top
    }

def gini_coefficient(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy()
    x = np.clip(x, 0, None)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    g = (n + 1 - 2 * (cum.sum() / cum[-1])) / n
    return float(g)

def _id_cols_for_level(df: pd.DataFrame, level: str) -> list[str]:
    if level == "district":
        return ["district", "state"]
    if level == "pin":
        # if state/district available, include them for context
        cols = []
        if "pincode" in df.columns:
            cols.append("pincode")
        if "district" in df.columns:
            cols.append("district")
        if "state" in df.columns:
            cols.append("state")
        return cols if cols else ["pincode"]
    raise ValueError("level must be 'district' or 'pin'")

def format_top_units(df: pd.DataFrame, level: str, topn: int = 5, value_col: str = "instability_score") -> str:
    top = df.sort_values(value_col, ascending=False).head(topn)
    id_cols = _id_cols_for_level(df, level)

    parts = []
    for _, r in top.iterrows():
        label = ", ".join([f"{c}={r[c]}" for c in id_cols if c in top.columns])
        parts.append(f"{label} → {r[value_col]:.1f}")
    return "; ".join(parts)

def reason_mix(df: pd.DataFrame, col: str = "reason_code")-> Optional[str]:
    if col not in df.columns:
        return None
    vc = df[col].value_counts(normalize=True).head(5) * 100
    return ", ".join([f"{k}:{v:.0f}%" for k, v in vc.items()])

def generate_dataset_insight(
    df_inst: pd.DataFrame,
    dataset_name: str,
    level: str = "district",            # "district" or "pin"
    x_list: list[float] = [0.05, 0.10, 0.20],
    topn: int = 5,
    value_col: str = "instability_score"
) -> str:
    if value_col not in df_inst.columns:
        raise ValueError(f"{dataset_name}: '{value_col}' column not found")

    # Check id columns exist
    if level == "district":
        required = {"state", "district"}
        missing = required - set(df_inst.columns)
        if missing:
            raise ValueError(f"{dataset_name}: missing required district columns: {missing}")
    elif level == "pin":
        if "pincode" not in df_inst.columns:
            raise ValueError(f"{dataset_name}: missing required pin column: 'pincode'")
    else:
        raise ValueError("level must be 'district' or 'pin'")

    g = gini_coefficient(df_inst[value_col])

    if g >= 0.50:
        conc_word = "highly concentrated"
    elif g >= 0.35:
        conc_word = "moderately concentrated"
    else:
        conc_word = "fairly distributed"

    conc_lines = []
    for x in x_list:
        res = top_x_contribution(df_inst, value_col=value_col, x=x)
        conc_lines.append(f"top {int(x*100)}% contribute {res['contribution_share_pct']}%")

    unit_word = "districts" if level == "district" else "PIN codes"
    conc_text = "; ".join(conc_lines)

    top_text = format_top_units(df_inst, level=level, topn=topn, value_col=value_col)

    mix = reason_mix(df_inst, col="reason_code")
    mix_text = f" Reason mix: {mix}." if mix else ""

    return (
        f"{dataset_name} ({unit_word}): Instability is {conc_word} (Gini={g:.2f}). "
        f"Concentration: {conc_text} of total instability. "
        f"Highest-risk {unit_word}: {top_text}.{mix_text}"
    )



# %% [markdown]
# ### **K Means**

# %% [markdown]
# We use K-means clustering here to discover natural groups of districts/PINs that behave similarly, rather than evaluating each location in isolation.
# 
# **What K-means adds beyond ranking or scoring?**
# 
# * Instability score ranks severity, but it does not explain why a district is unstable.
# * K-means clusters explain structure, grouping districts with similar instability drivers even if their total scores differ.
# 
# Two districts/pin-codes may have the same instability score:
# 
# * one driven by frequent small spikes,
# * another driven by a single extreme peak.
# 
# K-means separates them into different clusters, enabling tailored interventions.

# %%
import numpy as np
import pandas as pd


def kmeans_cluster_districts(
    df: pd.DataFrame,
    feature_cols: list[str],
    k: Optional[int] = None,
    k_min: int = 3,
    k_max: int = 8,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds a 'cluster' column using KMeans on feature_cols.
    If k=None, chooses best k by silhouette.
    Returns: (df_with_cluster, cluster_profile)
    """
    out = df.copy()
    X = out[feature_cols].fillna(0.0).to_numpy()

    # Standardize for KMeans fairness
    Xs = StandardScaler().fit_transform(X)

    if k is None:
        best_k, best_score = None, -1
        for kk in range(k_min, k_max + 1):
            model = KMeans(n_clusters=kk, n_init=30, random_state=random_state)
            labels = model.fit_predict(Xs)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(Xs, labels)
            if score > best_score:
                best_score, best_k = score, kk
        k = best_k if best_k is not None else 4

    kmeans = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    out["cluster"] = kmeans.fit_predict(Xs)

    # Cluster profile for interpretation
    profile = (
        out.groupby("cluster", as_index=False)
           .agg(
               n=("cluster", "size"),
               **{f"avg_{c}": (c, "mean") for c in feature_cols},
               avg_instability=("instability_score", "mean") if "instability_score" in out.columns else ("cluster","size")
           )
           .sort_values("avg_instability", ascending=False)
           .reset_index(drop=True)
    )

    return out, profile

def plot_clusters_pca(
    df: pd.DataFrame,
    feature_cols: list[str],
    title: str = "District Clusters (PCA projection)",
    hover_cols: Optional[list[str]] = None
):
    if hover_cols is None:
        hover_cols = ["state", "district", "instability_score", "reason_code"]

    plot_df = df.dropna(subset=feature_cols + ["cluster"]).copy()

    # keep only hover columns that exist
    hover_cols = [c for c in hover_cols if c in plot_df.columns]

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.express as px

    X = plot_df[feature_cols].fillna(0.0).to_numpy()
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    plot_df["pca_1"] = coords[:, 0]
    plot_df["pca_2"] = coords[:, 1]

    fig = px.scatter(
        plot_df,
        x="pca_1",
        y="pca_2",
        color="cluster",
        hover_data=hover_cols,
        title=title
    )
    fig.update_layout(height=650)
    fig.show()
    return fig


# %%
def plot_cluster_profile(profile_df: pd.DataFrame, feature_cols: list[str], title="Cluster Driver Profile"):
    # profile has avg_feature columns (avg_cov_norm etc.)
    cols = ["cluster", "n"] + [f"avg_{c}" for c in feature_cols]
    d = profile_df[cols].copy()

    long = d.melt(id_vars=["cluster","n"], var_name="metric", value_name="value")
    long["metric"] = long["metric"].str.replace("avg_", "", regex=False)

    fig = px.bar(long, x="metric", y="value", color="cluster", barmode="group",
                 title=title, hover_data=["n"])
    fig.update_layout(height=520)
    fig.show()
    return fig

# %% [markdown]
# **Plotting Enrollment District Instability**

# %%
features = ["cov_norm", "spike_norm", "peak_norm", "severity_norm"]  # drop severity_norm if not using it


# %% [markdown]
# 🔵 **Cluster 0 (Blue) — Stable / Low-Risk**
# 
# * Low across all metrics: cov_norm, spike_norm, peak_norm, severity_norm
# 
# What it represents:
# 
# *Districts with stable and predictable enrollment patterns, minimal spikes, and no extreme months.*
# 
# Operational meaning:
# 
# *These are baseline districts. Standard staffing and routine monitoring are sufficient.*
# 
# 🟣 **Cluster 1 (Purple) — Chronically Volatile**
# 
# * High volatility (cov_norm)
# * Moderate peak stress
# * Low-to-moderate spike frequency and severity
# 
# What it represents:
# 
# *Districts with persistent month-to-month fluctuations, but without sharp abnormal events.*
# 
# Operational meaning:
# 
# *Indicates structural instability in demand. These districts benefit from flexible staffing, smoothing policies, and better forecasting.*
# 
# 🌸 **Cluster 2 (Pink) — Mixed / High-Risk**
# 
# * Elevated volatility, peaks, and severity
# * Moderate spike frequency
# 
# What it represents:
# 
# *Districts experiencing multiple instability drivers simultaneously—chronic variability plus high-impact months.*
# 
# Operational meaning:
# 
# *This is the highest-risk cluster, requiring comprehensive intervention: monitoring, buffers, and contingency planning.*
# 
# 🟠 **Cluster 3 (Orange) — Peaky / Capacity-Shock Driven**
# 
# * Very high peak stress (peak_norm)
# * Moderate volatility
# * Low-to-moderate spike frequency
# 
# What it represents:
# 
# *Instability dominated by one or two extreme enrollment months, rather than frequent disruptions.*
# 
# Operational meaning:
# 
# *Best handled with temporary surge capacity during known drives or seasonal peaks.*
# 
# 🟡 **Cluster 4 (Yellow) — Spiky but Controlled**
# 
# * High spike frequency (spike_norm)
# * Lower peak stress and severity
# * Moderate volatility
# 
# What it represents:
# 
# *Districts with frequent enrollment surges, but each surge is relatively contained.*
# 
# Operational meaning:
# 
# *Requires early-warning systems and scheduling discipline, not heavy capacity expansion.*

# %%
features = ["cov_norm", "spike_norm", "peak_norm", "severity_norm"]  # drop severity_norm if not using it


# %% [markdown]
# ### **Reason Code**

# %% [markdown]
# * **VOLATILE**
#  
#     -Chronic month-to-month instability
# 
#     -CoV high relative to other districts
# 
#     -Interpretation: The district’s workload is structurally unstable (planning + staffing mismatch, inconsistent throughput, or irregular operations).
# 
# * **SPIKY**
# 
#     -Repeated abnormal events (drives, outages, backlog clearances).
# 
#     -spike months occur often across the district timeline
# 
#     -Interpretation: The district is usually stable, but repeatedly gets disrupted by episodic events.
# 
# * **PEAKY**
# 
#     -One or two extreme months causing capacity shock.
# 
#     -A single “mountain” month. Flat before and after. That one month explains most of the perceived instability.
# 
#     -Capacity planning problem, not chronic instability. You should plan temporary infrastructure / staff surge for known seasonal or campaign-like peak windows.
# 
# * **MIXED**
# 
#     -Multiple instability drivers → needs deeper investigation.
# 
#     -Both structural instability and event shocks exist; needs deeper investigation and multi-pronged intervention.
# 
# We decompose district instability into chronic variability (CoV), anomaly frequency (spike months), capacity shocks (peak factor), and spike intensity (severity). We assign a reason code only when one normalized driver clearly dominates; otherwise we label the district as MIXED to avoid false certainty and trigger deeper diagnostic review.

# %%
def add_reason_code_with_severity(
    df: pd.DataFrame,
    margin: float = 0.10,
    cov_col: str = "cov_norm",
    spike_col: str = "spike_norm",
    peak_col: str = "peak_norm",
    severity_col: str = "severity_norm",
    out_col: str = "reason_code",
) -> pd.DataFrame:
    """
    Assigns a dashboard-friendly reason code when Instability Score has 4 components:
      - VOLATILE  : cov_norm dominates
      - SPIKY     : spike_norm dominates (spike frequency)
      - PEAKY     : peak_norm dominates (peak factor)
      - INTENSE   : severity_norm dominates (spike intensity)
      - MIXED     : no clear dominant driver (top-2 within `margin`)

    Parameters
    ----------
    margin : float
        If (top - second) <= margin, mark as MIXED.
        Example: margin=0.10 means top driver must be >= 0.10 higher than second to "dominate".
    """
    out = df.copy()

    required = [cov_col, spike_col, peak_col, severity_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required normalized columns: {missing}")

    comps = out[required].to_numpy()

    # Top and second-top values per row (district)
    sorted_vals = np.sort(comps, axis=1)   # ascending
    top = sorted_vals[:, -1]
    second = sorted_vals[:, -2]

    # Which component is maximum: 0=cov, 1=spike, 2=peak, 3=severity
    argmax = np.argmax(comps, axis=1)

    # MIXED if top two are close
    is_mixed = (top - second) <= margin

    labels = np.array(["VOLATILE", "SPIKY", "PEAKY", "INTENSE"], dtype=object)

    out[out_col] = np.where(is_mixed, "MIXED", labels[argmax])

    # Optional: dominant component name for debugging/explainability
    dom_names = np.array(["cov", "spike_freq", "peak_factor", "severity"], dtype=object)
    out["dominant_component"] = np.where(is_mixed, "mixed", dom_names[argmax])

    return out


# %%
#Reason codes for instability at district level

# %% [markdown]
# ### **Forecast next month load**

# %% [markdown]
# Exponential Smoothing

# %%
def forecast_next_month_ewma(
    monthly_df: pd.DataFrame,
    group_keys: list[str],
    value_col: str = "total",
    alpha: float = 0.5
) -> pd.DataFrame:
    """
    One-step-ahead forecast using Simple Exponential Smoothing (EWMA).

    Forecast(t+1) = alpha * y(t) + (1 - alpha) * Forecast(t)
    Prediction interval is heuristic, based on EWMA std.
    """

    # Defensive copy & proper ordering
    df = monthly_df.copy().sort_values(group_keys + ["month"])
    g = df.groupby(group_keys, sort=False)

    # EWMA level (forecast signal)
    df["ewma"] = g[value_col].transform(
        lambda s: s.ewm(alpha=alpha, adjust=False).mean()
    )

    # EWMA-based volatility (uncertainty signal)
    df["ewm_std"] = g[value_col].transform(
        lambda s: s.ewm(alpha=alpha, adjust=False).std(bias=False)
    )

    # Take last observed month per group
    last = df.groupby(group_keys, as_index=False).tail(1).copy()

    # Forecast month = next calendar month
    last["forecast_month"] = (
        last["month"] + pd.offsets.MonthBegin(1)
    ).dt.to_period("M").dt.to_timestamp()

    # One-step-ahead forecast
    last["forecast"] = last["ewma"]

    # Heuristic 95% prediction interval
    last["pi_low"] = (last["forecast"] - 1.96 * last["ewm_std"].fillna(0)).clip(lower=0)
    last["pi_high"] = (last["forecast"] + 1.96 * last["ewm_std"].fillna(0)).clip(lower=0)

    cols = (
        group_keys
        + ["forecast_month", "forecast", "pi_low", "pi_high", "month", value_col]
    )

    return last[cols].rename(
        columns={"month": "last_month", value_col: "last_value"}
    )


# %% [markdown]
# ### **Top “consistent performers”**

# %%
def consistent_performers(monthly_df: pd.DataFrame, level: str = "district",
                          value_col: str = "total", min_months: int = 4,
                          volume_quantile: float = 0.6, topn: int = 50) -> pd.DataFrame:
    """
    Finds stable performers:
    - low CoV and low median MoM % change
    - but also above a volume threshold to avoid tiny-activity regions
    """
    if level == "district":
        keys = ["state","district"]
    elif level == "pin":
        keys = ["state","district","pincode"]
    else:
        raise ValueError("level must be 'district' or 'pin'")

    df = monthly_df.copy().sort_values(keys + ["month"])
    df["prev"] = df.groupby(keys)[value_col].shift(1)
    df["mom_abs_pct"] = np.where(df["prev"].fillna(0) == 0, np.nan, (df[value_col]-df["prev"]).abs()/df["prev"])

    agg = (
        df.groupby(keys, as_index=False)
          .agg(
              months=("month","nunique"),
              total_sum=(value_col,"sum"),
              mean=(value_col,"mean"),
              std=(value_col,"std"),
              median_mom_abs_pct=("mom_abs_pct","median"),
              max_total=(value_col,"max"),
          )
    )
    agg["cov"] = np.where(agg["mean"] == 0, np.nan, agg["std"]/agg["mean"])

    agg = agg[agg["months"] >= min_months].copy()
    vol_thresh = agg["total_sum"].quantile(volume_quantile)
    agg = agg[agg["total_sum"] >= vol_thresh].copy()

    # stability score: lower is better
    agg["stability_score"] = (agg["cov"].fillna(0) * 0.7) + (agg["median_mom_abs_pct"].fillna(0) * 0.3)
    agg = agg.sort_values(["stability_score","total_sum"], ascending=[True, False]).head(topn).reset_index(drop=True)
    return agg


