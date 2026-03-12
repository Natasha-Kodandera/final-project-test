import shutil
import subprocess
from pathlib import Path

import pytask

from cluster_analysis.config import BLD, DOCUMENTS, ROOT


@pytask.task(id="paper-pdf")
def task_compile_paper(
    paper_md: Path = DOCUMENTS / "paper.md",
    myst_yml: Path = ROOT / "myst.yml",
    refs: Path = DOCUMENTS / "refs.bib",
    silhouette_plot: Path = BLD / "plots" / "silhouette_scores.png",
    calinski_plot: Path = BLD / "plots" / "calinski_harabasz_scores.png",
    davies_plot: Path = BLD / "plots" / "davies_bouldin_scores.png",
    pca_plot: Path = BLD / "plots" / "pca_scatter_kmeans_5.png",
    produces: Path = ROOT / "paper.pdf",
) -> None:
    """Compile the paper from MyST Markdown to PDF."""
    subprocess.run(
        ("jupyter", "book", "build", "--pdf"),
        check=True,
        cwd=ROOT.resolve(),
    )

    build_pdf = ROOT / "_build" / "exports" / "paper.pdf"
    produces.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(build_pdf, produces)
