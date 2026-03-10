import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from cluster_analysis.analysis.clustering_model import fit_clustering_model
from cluster_analysis.config import RANDOM_STATE


def choose_n_clusters(
    data: pd.DataFrame,
    method: str,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate clustering model for cluster selection.

    Args:
    data: Prepared feature data used for clustering.
    method: Clustering method, either 'kmeans' or 'agglomerative'
    n_clusters: number of clusters to fit
    random_state: random seed used for kmeans

    Returns:
    One-row dataframe with clustering evaluation results.
    """
    model = fit_clustering_model(
        data=data,
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    labels = pd.Series(model.labels_)
    scores = _calculate_cluster_scores(data=data, labels=labels)
    cluster_sizes = labels.value_counts()

    out = pd.DataFrame(
        {
            "method": [method],
            "n_clusters": [n_clusters],
            "silhouette_score": [scores["silhouette_score"]],
            "calinski_harabasz_score": [scores["calinski_harabasz_score"]],
            "davies_bouldin_score": [scores["davies_bouldin_score"]],
            "inertia": [_calculate_inertia(model=model, method=method)],
            "min_cluster_size": [int(cluster_sizes.min())],
            "max_cluster_size": [int(cluster_sizes.max())],
        }
    )
    return out


def _calculate_cluster_scores(
    data: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:
    """Calculate internal clustering evaluation scores."""
    return {
        "silhouette_score": float(silhouette_score(data, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(data, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(data, labels)),
    }


def _calculate_inertia(model, method: str) -> float:
    """Return inertia for kmeans and NaN otherwise."""
    if method == "kmeans":
        return float(model.inertia_)
    return float("nan")
