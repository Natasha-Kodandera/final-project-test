import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

pio.templates.default = "plotly_white+presentation"

PROFILE_VARS = [
    "sample_share",
    "mean_age",
    "share_female",
    "share_hispanic",
    "share_employed",
    "mean_hours_weekly",
    "share_full_time_hours",
    "mean_earnings_hourly",
]


def plot_cluster_scores(
    df: pd.DataFrame,
    score_column: str,
    score_label: str,
) -> go.Figure:
    """Plot cluster quality scores by clustering method and number of clusters.

    Args:
        df (pd.DataFrame): Cluster selection score results.
        score_column (str): Column name of score to plot.
        score_label (str): Label for the score.

    Returns:
        go.Figure: Plotly figure.
    """
    plot_data = df.sort_values(["method", "n_clusters"])

    fig = px.line(
        plot_data,
        x="n_clusters",
        y=score_column,
        color="method",
        markers=True,
        labels={
            "n_clusters": "Number of clusters",
            score_column: score_label,
            "method": "Method",
        },
    )

    fig.update_layout(
        title=f"{score_label} by clustering method",
        legend_title_text="Method",
    )

    return fig


def plot_cluster_pca_scatter(
    df_features: pd.DataFrame,
    labels: pd.Series,
    model: str,
) -> go.Figure:
    """Plot clustered observations on the first 2 PCA components.

    Args:
        df_features (pd.DataFrame): prepared clustering feature matrix
        labels (pd.Series): cluster labels
        model (str): name of clustering method

    Returns:
        go.Figure: Plotly figure.
    """
    components = PCA(n_components=2).fit_transform(df_features)
    plot_data = pd.DataFrame(
        {
            "pc1": components[:, 0],
            "pc2": components[:, 1],
            "cluster": labels.astype(str),
        }
    )

    fig = px.scatter(
        plot_data,
        x="pc1",
        y="pc2",
        color="cluster",
        opacity=0.5,
        labels={
            "pc1": "Principal component 1",
            "pc2": "Principal component 2",
            "cluster": "Cluster",
        },
    )
    fig.update_layout(title=f"PCA scatter plot for {model}")
    return fig
