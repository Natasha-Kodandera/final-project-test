import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans


def fit_clustering_model(
    data: pd.DataFrame, method: str, n_clusters: int, random_state: int | None = None
) -> KMeans | AgglomerativeClustering:
    """Fit clustering model to prepared CPS data.

    Args:
        data: Feature data ready for clustering.
        method: Clustering method (kmeans or agglomerative)
        n_clusters: Number of clusters to fit
        random_state: Random seed used only for kmeans.

    Returns:
        Fitted clustering model.
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)

    else:
        msg = f"{method}: clustering method not supported."
        raise ValueError(msg)

    return model.fit(data)
