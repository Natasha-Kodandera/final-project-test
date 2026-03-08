import pandas as pd

from cluster_analysis.config import CATEGORICAL_VARS, CONTINUOUS_VARS

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

MIN_WORKING_AGE = 16
MISSING_CODES = {-1, -2, -3}
VALID_RANGE = {"age": (0, 85), "hours_weekly": (0, 99), "earnings_hourly": (0, 99.99)}


def clean_cps_data(raw: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Clean the CPS basic monthly dataset for January 2026.

    Args:
    raw (pd.DataFrame): Raw CPS dataframe.
    info (pd.DataFrame): The CPS variable info table

    Returns:
    pd.DataFrame: Cleaned CPS data with handled missing codes,
    enforced data types, and filtered for labour force.
    """
    df = pd.DataFrame(index=raw.index)

    cps_cols = info["cps_name"].tolist()
    clean_cols = info["readable_name"].tolist()

    for cps_col_name, clean_col_name in zip(cps_cols, clean_cols, strict=True):
        sr = pd.to_numeric(raw[cps_col_name], errors="coerce")
        sr = _replace_missing_codes(sr)

        if clean_col_name in CONTINUOUS_VARS:
            df[clean_col_name] = _clean_continuous(sr, clean_col_name)
        elif clean_col_name in CATEGORICAL_VARS:
            df[clean_col_name] = _clean_categorical(sr)
        else:
            df[clean_col_name] = sr

    df = _check_valid_range(df)
    df = _filter_labour_force(df)

    return df


def _replace_missing_codes(sr: pd.Series) -> pd.Series:
    """Replace negative values (CPS missing codes) with NA."""
    return sr.replace(list(MISSING_CODES), pd.NA)


def _clean_categorical(sr: pd.Series) -> pd.Series:
    """Convert variables to pandas categorical data type."""
    return sr.astype(pd.CategoricalDtype())


def _clean_continuous(sr: pd.Series, var: str) -> pd.Series:
    """Type casting for continuous variables and unit conversion (cents to dollars)."""
    sr = sr.copy()

    if var == "earnings_hourly":
        sr = sr / 100
    if var == "hours_weekly":
        sr = sr.replace(-4, pd.NA)

    return sr.astype(pd.Float32Dtype())


def _check_valid_range(df: pd.DataFrame) -> pd.DataFrame:
    """Check that values fall in valid range based on CPS data dictionary.

    Values outside the range converted to NA.
    """
    df = df.copy()

    for var, (min_value, max_value) in VALID_RANGE.items():
        if var in df.columns:
            invalid = (df[var] < min_value) | (df[var] > max_value)
            df.loc[invalid, var] = pd.NA

    return df


def _filter_labour_force(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only persons aged 16 years and above in the labour force."""
    labour_force = df["employment_status"].isin([1, 2])
    working_age = df["age"] >= MIN_WORKING_AGE
    filtered = df[labour_force & working_age].copy()
    return filtered
