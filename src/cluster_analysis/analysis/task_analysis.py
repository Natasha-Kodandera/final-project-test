import pickle
from pathlib import Path

import pandas as pd
import pytask

from cluster_analysis.analysis.cluster_selection import choose_n_clusters
from cluster_analysis.analysis.clustering_model import fit_clustering_model
from cluster_analysis.config import BLD, RANDOM_STATE, SRC

N_CLUSTERS: int = 2
N_CLUSTERS_TO_CHECK: tuple[int, ...] = (2, 3, 4, 5, 6)
METHODS: tuple[str, ...] = ("kmeans", "agglomerative")

for method in METHODS:

    @pytask.task(id=method)
    def task_fit_clustering_model(
        script: Path = SRC / "analysis" / "clustering_model.py",
        method: str = method,
        data: Path = BLD / "data" / "cps_clustering_data.feather",
        produces: Path = BLD / "model_results" / f"{method}.pkl",
    ) -> None:
        """Fit clustering model on the prepared CPS clustering data."""
        df = pd.read_feather(data)

        model = fit_clustering_model(
            data=df, method=method, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
        )

        produces.parent.mkdir(parents=True, exist_ok=True)
        with produces.open("wb") as file:
            pickle.dump(model, file)


def task_choose_n_clusters(
    script: Path = SRC / "analysis" / "cluster_selection.py",
    data: Path = BLD / "data" / "cps_clustering_data.feather",
    produces: Path = BLD / "model_results" / "cluster_selection_scores.feather",
) -> None:
    """Evaluate and choose from different numbers of clusters for clustering method."""
    df = pd.read_feather(data)

    results = []
    for method in METHODS:
        for n_clusters in N_CLUSTERS_TO_CHECK:
            result = choose_n_clusters(
                data=df,
                method=method,
                n_clusters=n_clusters,
                random_state=RANDOM_STATE,
            )
            results.append(result)

    produces.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(results, ignore_index=True)
    out.to_feather(produces)
