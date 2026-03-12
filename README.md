# Cluster Analysis on Labour Market Microdata

This project builds a reproducible pipeline implementing an unsupervised learning method
\- cluster analysis - using CPS labour market microdata to identify worker segments. The
pipeline supports k-means and agglomerative clustering, with final results produced
using k-means (5 clusters).

## Quick Start

1. Install [pixi](https://pixi.prefix.dev/latest/), which is the only prerequisite.
1. Clone the repository and navigate to the final project folder.
1. Install and activate the project environment:
   ```bash
   pixi install
   pixi shell
   ```
1. Build the project:
   ```bash
   #To run the full pipeline
   pixi run pytask

   #To run the tests
   pixi run pytest
   ```

## Directory Structure

- `src/cluster_analysis/` – Source code for the clustering pipeline.

  - `data/raw` – Zipped raw CPS data, data dictionary, and variable information.
  - `data_management/`
    - `clean_cps_data.py` – Cleans CPS raw data.
    - `prepare_clustering_data.py` – Prepares dataset for clustering.
    - `task_data_management.py` – Runs the data management pipeline.
  - `analysis/`
    - `cluster_selection.py` – Determines optimal number of clusters.
    - `clustered_data.py` – Assigns cluster labels to cleaned data.
    - `clustering_model.py` – Implements clustering algorithms.
    - `task_analysis.py` – Executes clustering analysis.
  - `final/`
    - `cluster_profiles.py` – Generates cluster summaries.
    - `plots.py` – Creates plots of clustering results.
    - `task_cluster_profiles.py` – Builds cluster profiles.
    - `task_plots.py` – Generates final plots.

- `tests/`

  - `analysis/` – Tests for clustering analysis code.
  - `data_management/` – Tests for data preparation code.

- `documents/`

  - `paper.md` – Final project report.
  - `task_paper.py` – Compiles the paper.

- `bld/` – Build directory with intermediate and final outputs (not version controlled).

  - `data/` – Raw unzipped and cleaned datasets.
  - `model_results/` – Saved clustering outputs.
  - `tables/` – Generated tables.
  - `plots/` – Generated figures.
  - `final/` – Final clustered dataset.
