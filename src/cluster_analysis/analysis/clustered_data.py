import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def create_clustered_data(
    cleaned_data: pd.DataFrame,
    model: KMeans | AgglomerativeClustering,
    cluster_column: str = "cluster",
) -> pd.DataFrame:
    """Add the cluster labels from fitted clustering model to cleaned CPS data.

    Args:
        cleaned_data: Cleaned CPS data in interpretable form.
        model: Fitted clustering model.
        cluster_column: Column that contains the cluster label.

    Returns:
        Cleaned CPS data with an additional column of cluster label.
    """
    _fail_if_input_not_dataframe(cleaned_data)
    _fail_if_model_no_labels(model)

    out = cleaned_data.copy()
    out[cluster_column] = pd.Series(model.labels_, index=out.index, dtype="Int64")
    out[cluster_column] = out[cluster_column].astype(pd.CategoricalDtype())

    return out


def _fail_if_input_not_dataframe(cleaned_data: pd.DataFrame) -> None:
    if not isinstance(cleaned_data, pd.DataFrame):
        msg = f"cleaned_data must be a pandas dataframe, not {type(cleaned_data)}."
        raise TypeError(msg)


def _fail_if_model_no_labels(model: KMeans | AgglomerativeClustering) -> None:
    if not hasattr(model, "labels_"):
        msg = "Fitted model must have labels_ attribute."
        raise ValueError(msg)
