# Cluster Analysis on Labour Market Microdata

+++ {"part": "abstract"}

+++

## Introduction

This project builds a reproducible pipeline to implement cluster analysis to identify
meaningful clusters or segmentation in the labour market. The workflow starts with the
raw data and proceeds to clean and preprocess the data, determine the clustering
specifications and optimal number of clusters, and finally produces tables and figures
for easy visualisation of the results.

The pipeline is designed to support both k-means and agglomerative clustering. Although
the final results only include the k-means model as it yields more interpretable
results, the code can incorporate agglomerative clustering by simply adding it to the
model dictionary, without having to restructure the entire codebase.

## Data

The raw data used is the Current Population Survey (CPS) basic monthly file for January
2026, released by the US Census Bureau. A mix of demographic variables and labour market
indicators are chosen - such as age, sex, employment status, hours worked per week,
earnings per hour, class of worker, industry, occupation and so on. These variables of
interest are also listed in the `variable_info` file.

The sample for clustering is restricted to working-age population who are in the labour
force. The variables considered for clustering include `age`, `sex`,
`employment status`, `hours worked weekly` and `earnings per hour`.

## Workflow

1. Raw data cleaning
1. Preprocessing data to make it ready for clustering
1. Selecting optimal number of clusters
1. Clustering model estimated and evaluated
1. Generating cluster profiles and plots

## Results

Both k-means and agglomerative clustering run through the same pipeline. Cluster quality
is compared using different standard scores and based on these metrics, I choose k-means
with 5 clusters as the final model specification to include in the results, owing to
interpretability and runtime considerations.

```{figure} ../bld/plots/silhouette_scores.png
---
width: 85%
label: fig:silhouette
---
Higher Silhouette score indicates well-separated clusters.
```

```{figure} ../bld/plots/calinski_harabasz_scores.png
---
width: 85%
label: fig:calinski_harabasz
---
Higher Calinski-Harabasz scores are better as it indicates more separation and more compact clusters.
```

```{figure} ../bld/plots/davies_bouldin_scores.png
---
width: 85%
label: fig:davies_bouldin
---
Lower Davies-Bouldin score is better.
```

```{figure} ../bld/plots/pca_scatter_kmeans_5.png
---
width: 85%
label: fig:pca_scatter
---
Cluster assignments visualised in PCA-reduced dimensions for k-means model with 5 clusters.
```

The k-means clustering identifies five labour market segments showing these patterns in
demographics and earnings (also suggested by the cluster profiles generated in the
pipeline) -

- Cluster 0: Mostly female part-time workers
- Cluster 1: Older (higher mean age) hispanic-dominated group
- Cluster 2: Younger (relatively lower mean age), full-time workers
- Cluster 3: Female-dominated older workers
- Cluster 4: High earning people, but the sample is quite small

The PCA scatter also suggests heterogeneity and general cluster separation, but there is
also substantial overlap in the centre, so the interpretation is not quite clear here.
