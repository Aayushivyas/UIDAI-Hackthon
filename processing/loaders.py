import pandas as pd


# -----------------------------
# Common helpers
# -----------------------------

def _standardize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns shared across all datasets.
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["state"] = df["state"].astype(str).str.strip().str.title()
    df["district"] = df["district"].astype(str).str.strip()
    df["pincode"] = df["pincode"].astype(str).str.strip()

    return df


def _melt_age_columns(
    df: pd.DataFrame,
    age_columns: list[str],
    transaction_type: str
) -> pd.DataFrame:
    """
    Convert wide age columns into long transactional format.
    """
    df_long = df.melt(
        id_vars=["date", "state", "district", "pincode"],
        value_vars=age_columns,
        var_name="age_group",
        value_name="count"
    )

    df_long["transaction_type"] = transaction_type

    return df_long


# -----------------------------
# Enrollment loader
# -----------------------------

def load_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
    date | state | district | pincode | age_0_5 | age_5_17 | age_18_greater
    """
    df = _standardize_common_fields(df)

    age_columns = [
        "age_0_5",
        "age_5_17",
        "age_18_greater"
    ]

    return _melt_age_columns(
        df=df,
        age_columns=age_columns,
        transaction_type="enrolment"
    )


# -----------------------------
# Demographic update loader
# -----------------------------

def load_demographic_updates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
    date | state | district | pincode | demo_age_5_17 | demo_age_17_
    """
    df = _standardize_common_fields(df)

    age_columns = [
        "demo_age_5_17",
        "demo_age_17_"
    ]

    return _melt_age_columns(
        df=df,
        age_columns=age_columns,
        transaction_type="demographic_update"
    )


# -----------------------------
# Biometric update loader
# -----------------------------

def load_biometric_updates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected columns:
    date | state | district | pincode | bio_age_5_17 | bio_age_17_
    """
    df = _standardize_common_fields(df)

    age_columns = [
        "bio_age_5_17",
        "bio_age_17_"
    ]

    return _melt_age_columns(
        df=df,
        age_columns=age_columns,
        transaction_type="biometric_update"
    )


# -----------------------------
# Unified fact table builder
# -----------------------------

def build_transaction_fact(
    enrollment_df: pd.DataFrame,
    demographic_df: pd.DataFrame,
    biometric_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Produces the single source of truth fact table.
    """
    fact_df = pd.concat(
        [
            load_enrollment(enrollment_df),
            load_demographic_updates(demographic_df),
            load_biometric_updates(biometric_df)
        ],
        ignore_index=True
    )

    # Ensure count is numeric and safe
    fact_df["count"] = pd.to_numeric(
        fact_df["count"],
        errors="coerce"
    ).fillna(0)

    return fact_df
