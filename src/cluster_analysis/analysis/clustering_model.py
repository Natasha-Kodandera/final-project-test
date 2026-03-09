import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans


def fit_clustering_model(
    data: pd.DataFrame,
    method: str,
    n_clusters: int,
):
    """Fit clustering model to CPS monthly data.

    Args:
    data(pd.DataFrame): Feature data ready for clustering.
    method(str): Clustering method (K-means or agglomerative)
    n_clusters(int): number of clusters

    Returns:
    Fitted clustering model.
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=123, n_init="auto")
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)

    else:
        msg = "method should be 'kmeans' or 'agglomerative'."
        raise ValueError(msg)

    return model.fit(data)
