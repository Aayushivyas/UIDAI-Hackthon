import pandas as pd


def validate_schema(df: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "state",
        "district",
        "pincode",
        "transaction_type",
        "age_group",
        "count"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_data_types(df: pd.DataFrame) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("Column 'date' must be datetime")

    if not pd.api.types.is_numeric_dtype(df["count"]):
        raise TypeError("Column 'count' must be numeric")


def validate_transaction_types(df: pd.DataFrame) -> None:
    allowed = {
        "enrolment",
        "demographic_update",
        "biometric_update"
    }

    bad = set(df["transaction_type"].unique()) - allowed
    if bad:
        raise ValueError(f"Invalid transaction_type values: {bad}")


def validate_non_negative_counts(df: pd.DataFrame) -> None:
    if (df["count"] < 0).any():
        raise ValueError("Negative transaction counts detected")


def validate_nulls(df: pd.DataFrame) -> None:
    critical = [
        "date",
        "state",
        "district",
        "transaction_type",
        "age_group",
        "count"
    ]

    nulls = df[critical].isna().any()
    bad_cols = nulls[nulls].index.tolist()

    if bad_cols:
        raise ValueError(f"Null values detected in columns: {bad_cols}")


def validate_pincode(df: pd.DataFrame) -> None:
    if not df["pincode"].astype(str).str.match(r"^\d{6}$").all():
        raise ValueError("Invalid pincode format detected")


def validate_duplicates(df: pd.DataFrame) -> None:
    dup_cols = [
        "date",
        "state",
        "district",
        "pincode",
        "transaction_type",
        "age_group"
    ]

    if df.duplicated(subset=dup_cols).any():
        raise ValueError("Duplicate transactional rows detected")


def validate_age_group_presence(df: pd.DataFrame) -> None:
    """
    Only ensures age_group exists and is usable.
    Naming rules are handled later.
    """
    if df["age_group"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty age_group values detected")


def run_all_validations(df: pd.DataFrame) -> None:
    validate_schema(df)
    validate_data_types(df)
    validate_transaction_types(df)
    validate_non_negative_counts(df)
    validate_nulls(df)
    validate_pincode(df)
    validate_age_group_presence(df)
    validate_duplicates(df)
