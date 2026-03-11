import pickle
from pathlib import Path

import pandas as pd
import pytask

from cluster_analysis.analysis.cluster_selection import choose_n_clusters
from cluster_analysis.analysis.clustered_data import create_clustered_data
from cluster_analysis.analysis.clustering_model import fit_clustering_model
from cluster_analysis.config import BLD, RANDOM_STATE, SRC

FINAL_N_CLUSTERS: dict[str, int] = {"kmeans": 2, "agglomerative": 4}
N_CLUSTERS_TO_CHECK: tuple[int, ...] = (2, 3, 4, 5, 6)
METHODS: tuple[str, ...] = ("kmeans", "agglomerative")

for method, n_clusters in FINAL_N_CLUSTERS.items():

    @pytask.task(id=f"{method}_{n_clusters}")
    def task_fit_clustering_model(
        script: Path = SRC / "analysis" / "clustering_model.py",
        method: str = method,
        n_clusters: int = n_clusters,
        data: Path = BLD / "data" / "cps_clustering_data.feather",
        produces: Path = BLD / "model_results" / f"{method}_{n_clusters}.pkl",
    ) -> None:
        """Fit clustering model on the prepared CPS clustering data."""
        df = pd.read_feather(data)

        model = fit_clustering_model(
            data=df, method=method, n_clusters=n_clusters, random_state=RANDOM_STATE
        )

        produces.parent.mkdir(parents=True, exist_ok=True)
        with produces.open("wb") as file:
            pickle.dump(model, file)


def task_choose_n_clusters(
    script: Path = SRC / "analysis" / "cluster_selection.py",
    data: Path = BLD / "data" / "cps_clustering_data.feather",
    produces: Path = BLD / "model_results" / "cluster_selection_scores.feather",
) -> None:
    """Evaluate and choose from different numbers of clusters for clustering methods."""
    df = pd.read_feather(data)

    results = []
    for method in METHODS:
        for n_clusters_check in N_CLUSTERS_TO_CHECK:
            result = choose_n_clusters(
                data=df,
                method=method,
                n_clusters=n_clusters_check,
                random_state=RANDOM_STATE,
            )
            results.append(result)

    produces.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(results, ignore_index=True)
    out.to_feather(produces)


for method, n_clusters in FINAL_N_CLUSTERS.items():

    @pytask.task(id=f"{method}_{n_clusters}")
    def task_create_clustered_data(
        script: Path = SRC / "analysis" / "clustered_data.py",
        method: str = method,
        n_clusters: int = n_clusters,
        data: Path = BLD / "data" / "cps_cleaned.feather",
        model_file: Path = BLD / "model_results" / f"{method}_{n_clusters}.pkl",
        produces: Path = BLD / "final" / f"cps_clustered_{method}_{n_clusters}.feather",
    ) -> None:
        """Attach cluster labels to the cleaned CPS data to make it interpretable."""
        df = pd.read_feather(data)

        with model_file.open("rb") as file:
            model = pickle.load(file)

        out = create_clustered_data(cleaned_data=df, model=model)
        produces.parent.mkdir(parents=True, exist_ok=True)
        out.to_feather(produces)
