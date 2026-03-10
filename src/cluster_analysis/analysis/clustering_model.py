import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans


def fit_clustering_model(
    data: pd.DataFrame,
    method: str,
    n_clusters: int,
    random_state: int | None = None,
):
    """Fit clustering model to prepared CPS data.

    Args:
    data(pd.DataFrame): Feature data ready for clustering.
    method(str): Clustering method (kmeans or agglomerative)
    n_clusters(int): number of clusters to fit
    random_state(int): random seed used only for kmeans

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
