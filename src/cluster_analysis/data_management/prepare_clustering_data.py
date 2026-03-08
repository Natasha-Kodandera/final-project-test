import pandas as pd

from cluster_analysis.config import CATEGORICAL_VARS, CONTINUOUS_VARS

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def prepare_clustering_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for clustering."""
    out = _select_clustering_features(df)
    return out


def _select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only features used for clustering."""
    feature_columns = CONTINUOUS_VARS + CATEGORICAL_VARS
    data_columns = [column for column in feature_columns if column in df.columns]
    return df[data_columns].copy()
