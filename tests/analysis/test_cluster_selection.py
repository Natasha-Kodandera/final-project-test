import math

import pandas as pd
import pytest
from sklearn.cluster import KMeans

from cluster_analysis.analysis.cluster_selection import (
    _calculate_inertia,
    choose_n_clusters,
)


@pytest.fixture
def sample_cluster_data() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x": [1, 2, 100, 101],
            "y": [1, 2, 100, 101],
        }
    )
    return df


def test_calculate_inertia_gives_nan_agglomerative(
    sample_cluster_data: pd.DataFrame,
) -> None:
    model = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(sample_cluster_data)
    result = _calculate_inertia(model=model, method="agglomerative")
    assert math.isnan(result)


def test_choose_n_clusters_validate_output_columns(
    sample_cluster_data: pd.DataFrame,
) -> None:
    result = choose_n_clusters(data=sample_cluster_data, method="kmeans", n_clusters=2)
    exp = pd.DataFrame(
        {
            "method": ["kmeans"],
            "n_clusters": [2],
            "min_cluster_size": [2],
            "max_cluster_size": [2],
        }
    )

    pd.testing.assert_frame_equal(
        result[["method", "n_clusters", "min_cluster_size", "max_cluster_size"]],
        exp,
    )
