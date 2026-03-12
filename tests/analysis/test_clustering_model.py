import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans

from cluster_analysis.analysis.clustering_model import fit_clustering_model


def test_fit_clustering_model_kmeans() -> None:
    df = pd.DataFrame({"x": [1, 3, 10, 13], "y": [1, 3, 10, 13]})
    result = fit_clustering_model(data=df, method="kmeans", n_clusters=2)
    assert isinstance(result, KMeans)


def test_fit_clustering_model_agglomerative() -> None:
    df = pd.DataFrame({"x": [1, 3, 10, 13], "y": [1, 3, 10, 13]})
    result = fit_clustering_model(data=df, method="agglomerative", n_clusters=2)
    assert isinstance(result, AgglomerativeClustering)


def test_fit_clustering_model_invalid_model() -> None:
    df = pd.DataFrame({"x": [1, 3], "y": [1, 3]})
    with pytest.raises(ValueError, match="not supported"):
        fit_clustering_model(data=df, method="invalid", n_clusters=2)
