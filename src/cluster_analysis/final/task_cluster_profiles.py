from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from cluster_analysis.config import BLD, SRC
from cluster_analysis.final.cluster_profiles import create_cluster_profiles

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


FINAL_CLUSTERED_DATA: dict[str, Path] = {
    "kmeans_5": BLD / "final" / "cps_clustered_kmeans_5.feather",
}

for model, data in FINAL_CLUSTERED_DATA.items():

    @pytask.task(id=model)
    def task_create_cluster_profiles(
        script: Path = SRC / "final" / "cluster_profiles.py",
        model: str = model,
        data: Path = data,
        produces_feather: Annotated[Path, Product] = BLD
        / "final"
        / f"cluster_profiles_{model}.feather",
        produces_markdown: Annotated[Path, Product] = BLD
        / "tables"
        / f"cluster_profiles_{model}.md",
    ) -> None:
        """Create cluster profile table for final clustered CPS dataset."""
        df = pd.read_feather(data)
        out = create_cluster_profiles(df)

        produces_feather.parent.mkdir(parents=True, exist_ok=True)
        out.to_feather(produces_feather)

        table = out.copy()

        numeric_cols = table.select_dtypes(include="number").columns
        table[numeric_cols] = table[numeric_cols].round(3)

        table = table.T
        table.columns = [f"Cluster {col}" for col in table.columns]

        for char in ("|", "[", "]"):
            escaped = f"\\{char}"
            table.columns = pd.Index(
                [str(col).replace(char, escaped) for col in table.columns]
            )

        produces_markdown.parent.mkdir(parents=True, exist_ok=True)
        with produces_markdown.open("w") as f:
            f.write(table.to_markdown(index=True))
