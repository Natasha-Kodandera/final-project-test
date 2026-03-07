import pandas as pd

from cluster_analysis.config import CATEGORICAL_VARS, CONTINUOUS_VARS

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

MIN_WORKING_AGE = 16


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

    df = _filter_labour_force(df)

    return df


def _replace_missing_codes(sr: pd.Series) -> pd.Series:
    """Replace negative values (CPS missing codes) with NA."""
    return sr.where(sr >= 0, other=pd.NA)


def _clean_categorical(sr: pd.Series) -> pd.Series:
    """Convert variables to pandas categorical data type."""
    return sr.astype(pd.CategoricalDtype())


def _clean_continuous(sr: pd.Series, var: str) -> pd.Series:
    """Type casting for continuous variables and unit conversion (cents to dollars)."""
    sr = sr.copy()

    if var == "earnings_hourly":
        sr = sr / 100.0
    return sr.astype(pd.Float32Dtype())


def _filter_labour_force(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only persons aged 16 years and above in the labour force."""
    labour_force = df["employment_status"].isin([1, 2])
    working_age = df["age"] >= MIN_WORKING_AGE
    filtered = df[labour_force & working_age].copy()
    return filtered
