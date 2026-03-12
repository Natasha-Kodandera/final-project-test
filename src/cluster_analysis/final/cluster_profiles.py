import pandas as pd

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True


def create_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Create table of cluster profiles from clustered CPS data.

    Args:
        df(pd.DataFrame): Cleaned CPS data with cluster labels.

    Returns:
        pd.DataFrame: Cluster-level summary table.
    """
    grouped = df.groupby("cluster", observed=True, sort=True)

    out = pd.DataFrame(index=grouped.size().index)
    out["n_obs"] = grouped.size()
    out["sample_share"] = out["n_obs"] / len(df)
    out["mean_age"] = grouped["age"].apply(_compute_mean)
    out["share_female"] = grouped["sex"].apply(lambda sr: _compute_share_equals(sr, 2))
    out["share_hispanic"] = grouped["hispanic"].apply(
        lambda sr: _compute_share_equals(sr, 1)
    )
    out["share_employed"] = grouped["employment_status"].apply(
        lambda sr: _compute_share_equals(sr, 1)
    )
    out["mean_hours_weekly"] = grouped["hours_weekly"].apply(_compute_mean)
    out["share_full_time_hours"] = grouped["hours_weekly"].apply(
        lambda sr: _compute_share_greater_equals(sr, 35)
    )
    out["mean_earnings_hourly"] = grouped["earnings_hourly"].apply(_compute_mean)

    out = out.reset_index()
    return out


def _compute_mean(sr: pd.Series) -> float:
    """Compute mean of series, ignoring missing values."""
    series = pd.to_numeric(sr).dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def _compute_share_equals(sr: pd.Series, value: int) -> float:
    """Compute share of non-missing observations equal to given value."""
    series = pd.to_numeric(sr).dropna()
    if series.empty:
        return float("nan")
    return float((series == value).mean())


def _compute_share_greater_equals(sr: pd.Series, value: int) -> float:
    """Compute share of observations greater than or equal to given value."""
    series = pd.to_numeric(sr).dropna()
    if series.empty:
        return float("nan")
    return float((series >= value).mean())
