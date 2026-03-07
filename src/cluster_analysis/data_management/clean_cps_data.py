import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

CONTINUOUS_VARS = {"age", "hours_weekly", "earnings_hourly"}
CATEGORICAL_VARS = {
    "sex",
    "education_level",
    "race",
    "hispanic",
    "employment_status",
    "full_part_time",
    "class_of_worker",
    "industry",
    "occupation",
}


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
