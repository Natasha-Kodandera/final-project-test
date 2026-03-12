from pathlib import Path

import pandas as pd
import pytask

from cluster_analysis.config import BLD, SRC
from cluster_analysis.final.plots import (
    plot_cluster_pca_scatter,
    plot_silhouette_scores,
)

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


FINAL_CLUSTERED_DATA: dict[str, Path] = {
    "kmeans_2": BLD / "final" / "cps_clustered_kmeans_2.feather",
    "agglomerative_4": BLD / "final" / "cps_clustered_agglomerative_4.feather",
}


def task_plot_silhouette_scores(
    script: Path = SRC / "final" / "plots.py",
    data: Path = BLD / "model_results" / "cluster_selection_scores.feather",
    produces: Path = BLD / "plots" / "silhouette_scores.png",
) -> None:
    """Plot silhouette scores by clustering method."""
    df = pd.read_feather(data)
    fig = plot_silhouette_scores(df)

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
