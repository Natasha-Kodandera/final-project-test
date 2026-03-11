from pathlib import Path

import pandas as pd
import pytask

from cluster_analysis.config import BLD, SRC
from cluster_analysis.final.cluster_profiles import create_cluster_profiles

FINAL_CLUSTERED_DATA: dict[str, Path] = {
    "kmeans_2": BLD / "final" / "cps_clustered_kmeans_2.feather",
    "agglomerative_4": BLD / "final" / "cps_clustered_agglomerative_4.feather",
}

for model, data in FINAL_CLUSTERED_DATA.items():

    @pytask.task(id=model)
    def task_create_cluster_profiles(
        script: Path = SRC / "final" / "cluster_profiles.py",
        model: str = model,
        data: Path = data,
        produces: Path = BLD / "final" / f"cluster_profiles_{model}.feather",
    ) -> None:
        """Create cluster profile table for final clustered CPS dataset."""
        df = pd.read_feather(data)
        out = create_cluster_profiles(df)

        produces.parent.mkdir(parents=True, exist_ok=True)
        out.to_feather(produces)
