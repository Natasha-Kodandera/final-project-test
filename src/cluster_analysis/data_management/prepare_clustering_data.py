import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cluster_analysis.config import CATEGORICAL_VARS, CONTINUOUS_VARS

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def prepare_clustering_data(df: pd.DataFrame) -> pd.DataFrame:
    """Make cleaned CPS data suitable for clustering.

    Perform preprocessing steps required by distance-based clustering algorithms:
    1. Selecting variables/features relevant for clustering
    2. Impute missing values
    3. Apply log transformation to skewed variables
    4. Convert categorical variables to dummies
    5. Standardize continuous variables

    Args:
    df (pd.DataFrame): Clean CPS dataset produced by data cleaning pipeline.

    Returns:
    pd.DataFrame: Fully numeric feature matrix suitable for clustering.
    """
    _fail_if_input_not_dataframe(df)
    _fail_if_earnings_negative(df)

    out = _select_clustering_features(df)
    out = _impute_missing_values(out)
    out = _transform_log(out)
    out = _convert_categorical_to_dummy(out)
    out = _standardize_continuous_variables(out)

    return out


def _select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only features used for clustering."""
    feature_columns = CONTINUOUS_VARS + CATEGORICAL_VARS
    data_columns = [column for column in feature_columns if column in df.columns]

    return df[data_columns].copy()


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values.

    Median for continuous features, and mode for categorical features.
    """
    out = df.copy()

    for column in CONTINUOUS_VARS:
        if column in out.columns and out[column].isna().any():
            median_value = out[column].median(skipna=True)
            out[column] = out[column].fillna(median_value)

    for column in CATEGORICAL_VARS:
        if column in out.columns and out[column].isna().any():
            mode_value = out[column].mode(dropna=True).iloc[0]
            out[column] = out[column].fillna(mode_value)

    return out


def _transform_log(df: pd.DataFrame) -> pd.DataFrame:
    """Log transformation of highly skewed continuous variable."""
    out = df.copy()

    if "earnings_hourly" in out.columns:
        out["earnings_hourly"] = np.log1p(out["earnings_hourly"])

    return out


def _convert_categorical_to_dummy(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical features to dummies."""
    out = df.copy()
    columns_to_convert = [
        column for column in CATEGORICAL_VARS if column in out.columns
    ]

    if not columns_to_convert:
        return out

    out = pd.get_dummies(out, columns=columns_to_convert, drop_first=False, dtype=float)

    return out


def _standardize_continuous_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Standardization of continuous variables in clustering features."""
    out = df.copy()
    columns_to_standardize = [
        column for column in CONTINUOUS_VARS if column in out.columns
    ]

    if not columns_to_standardize:
        return out

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(out[columns_to_standardize])
    out[columns_to_standardize] = scaled_values

    return out


def _fail_if_input_not_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        msg = f"Input must be a pandas dataframe, not {type(df)}."
        raise TypeError(msg)


def _fail_if_earnings_negative(df: pd.DataFrame) -> None:
    if "earnings_hourly" in df.columns and (df["earnings_hourly"] < 0).any():
        msg = "earnings_hourly cannot be negative."
        raise ValueError(msg)
