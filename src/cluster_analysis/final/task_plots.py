from pathlib import Path

import pandas as pd
import pytask

from cluster_analysis.config import BLD, SRC
from cluster_analysis.final.plots import (
    plot_cluster_pca_scatter,
    plot_cluster_scores,
)

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

SCORE_SPECS: dict[str, dict[str, str]] = {
    "silhouette": {
        "column": "silhouette_score",
        "label": "Silhouette score",
        "file": "silhouette_scores.png",
    },
    "calinski_harabasz": {
        "column": "calinski_harabasz_score",
        "label": "Calinski-Harabasz score",
        "file": "calinski_harabasz_scores.png",
    },
    "davies_bouldin": {
        "column": "davies_bouldin_score",
        "label": "Davies-Bouldin score",
        "file": "davies_bouldin_scores.png",
    },
}


FINAL_CLUSTERED_DATA: dict[str, Path] = {
    "kmeans_5": BLD / "final" / "cps_clustered_kmeans_5.feather",
}


for score_name, spec in SCORE_SPECS.items():

    @pytask.task(id=score_name)
    def task_plot_cluster_scores(
        script: Path = SRC / "final" / "plots.py",
        score_column: str = spec["column"],
        score_label: str = spec["label"],
        data: Path = BLD / "model_results" / "cluster_selection_scores.feather",
        produces: Path = BLD / "plots" / spec["file"],
    ) -> None:
        """Plot cluster-quality scores by clustering method."""
        df = pd.read_feather(data)

        fig = plot_cluster_scores(
            df=df,
            score_column=score_column,
            score_label=score_label,
        )

        produces.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(produces)


for model, data in FINAL_CLUSTERED_DATA.items():

    @pytask.task(id=f"pca_{model}")
    def task_plot_cluster_pca_scatter(
        script: Path = SRC / "final" / "plots.py",
        model: str = model,
        features: Path = BLD / "data" / "cps_clustering_data.feather",
        data: Path = data,
        produces: Path = BLD / "plots" / f"pca_scatter_{model}.png",
    ) -> None:
        """Plot PCA scatter for clustered CPS data."""
        df_features = pd.read_feather(features)
        df_clustered = pd.read_feather(data)

        fig = plot_cluster_pca_scatter(
            df_features=df_features,
            labels=df_clustered["cluster"],
            model=model,
        )

        produces.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(produces)
