import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cluster_analysis.data_management.prepare_clustering_data import (
    _convert_categorical_to_dummy,
    _impute_missing_values,
    _select_clustering_features,
    prepare_clustering_data,
)


@pytest.fixture
def clean_data() -> pd.DataFrame:
    data = {
        "age": pd.Series([20, 40, 60, 70], dtype="Float32"),
        "hours_weekly": pd.Series([20, pd.NA, 80, 40], dtype="Float32"),
        "earnings_hourly": pd.Series([15, pd.NA, 25, 50], dtype="Float32"),
        "sex": pd.Series(pd.Categorical([1, 1, 2, 2], categories=[1, 2])),
        "employment_status": pd.Series(
            pd.Categorical([1, pd.NA, 1, 2], categories=[1, 2])
        ),
        "industry": pd.Series(
            pd.Categorical([10, 12, 14, 7], categories=[7, 10, 12, 14])
        ),
    }
    return pd.DataFrame(data)


def test_select_clustering_features_keeps_only_selected_features(
    clean_data: pd.DataFrame,
) -> None:
    got = _select_clustering_features(clean_data)
    exp = clean_data[
        [
            "age",
            "hours_weekly",
            "earnings_hourly",
            "sex",
            "employment_status",
        ]
    ].copy()
    assert_frame_equal(got, exp, check_like=True)


def test_impute_missing_values_replaces_na(clean_data: pd.DataFrame) -> None:
    got = _impute_missing_values(clean_data)
    exp = clean_data.copy()
    exp["hours_weekly"] = exp["hours_weekly"].fillna(40)
    exp["earnings_hourly"] = exp["earnings_hourly"].fillna(25)
    exp["employment_status"] = exp["employment_status"].fillna(1)
    assert_frame_equal(got, exp)


def test_convert_categorical_to_dummy() -> None:
    df = pd.DataFrame({"sex": pd.Series(pd.Categorical([1, 1, 2], categories=[1, 2]))})
    got = _convert_categorical_to_dummy(df)
    exp = pd.DataFrame(
        {
            "sex_1": pd.Series([1.0, 1.0, 0.0]),
            "sex_2": pd.Series([0.0, 0.0, 1.0]),
        }
    )
    assert_frame_equal(got, exp)


def test_prepare_clustering_data_no_missing_values(clean_data: pd.DataFrame) -> None:
    got = prepare_clustering_data(clean_data)
    assert got.isna().sum().sum() == 0


def test_prepare_clustering_data_has_numeric_cols(clean_data: pd.DataFrame) -> None:
    got = prepare_clustering_data(clean_data)
    assert got.select_dtypes(exclude="number").empty


def test_prepare_clustering_data_raises_error_on_negative_earnings() -> None:
    df = pd.DataFrame({"earnings_hourly": pd.Series([150, -500, 300], dtype="Float32")})
    with pytest.raises(ValueError, match="cannot be negative"):
        prepare_clustering_data(df)
