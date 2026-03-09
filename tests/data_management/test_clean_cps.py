import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from cluster_analysis.data_management.clean_cps_data import (
    _check_valid_range,
    _clean_categorical,
    _clean_continuous,
    _filter_labour_force,
    _replace_missing_codes,
    clean_cps_data,
)


@pytest.fixture
def raw_data() -> pd.DataFrame:
    data = {
        "prtage": [14, 30, 45, 87],
        "pesex": [1, 2, 1, 2],
        "peeduca": [40, 41, 43, 38],
        "ptdtrace": [4, 5, 6, 1],
        "pehspnon": [1, 2, 1, 2],
        "prempnot": [1, 2, 3, 4],
        "prwkstat": [1, 5, 7, 11],
        "pehrusl1": [-4, 50, 60, 120],
        "prcow1": [1, 2, 3, 4],
        "prmjind1": [10, 20, 30, 40],
        "prmjocc1": [5, 7, 8, 11],
        "pternhly": [2500, 1500, 10000, -50],
    }
    return pd.DataFrame(data)


@pytest.fixture
def info() -> pd.DataFrame:
    data = {
        "cps_name": [
            "prtage",
            "pesex",
            "peeduca",
            "ptdtrace",
            "pehspnon",
            "prempnot",
            "prwkstat",
            "pehrusl1",
            "prcow1",
            "prmjind1",
            "prmjocc1",
            "pternhly",
        ],
        "readable_name": [
            "age",
            "sex",
            "education_level",
            "race",
            "hispanic",
            "employment_status",
            "full_part_time",
            "hours_weekly",
            "class_of_worker",
            "industry",
            "occupation",
            "earnings_hourly",
        ],
    }
    return pd.DataFrame(data)


def test_replace_missing_codes_convert_to_na() -> None:
    sr = pd.Series([-1, -2, -3, 7])
    exp = pd.Series([pd.NA, pd.NA, pd.NA, 7])
    got = _replace_missing_codes(sr)
    assert_series_equal(got, exp)


def test_clean_categorical_dtype() -> None:
    sr = pd.Series([1, 2, 1])
    got = _clean_categorical(sr)
    assert str(got.dtype) == "category"


def test_clean_continuous_unit_conversion() -> None:
    sr = pd.Series([4500, 1000])
    exp = pd.Series([45.0, 10.0], dtype="Float32")
    got = _clean_continuous(sr, "earnings_hourly")
    assert_series_equal(got, exp)


def test_clean_continuous_invalid_hours() -> None:
    sr = pd.Series([20, -4, 40])
    exp = pd.Series([20.0, pd.NA, 40.0], dtype="Float32")
    got = _clean_continuous(sr, "hours_weekly")
    assert_series_equal(got, exp)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("age", 100),
        ("hours_weekly", 150),
        ("earnings_hourly", 1000),
    ],
)
def test_check_valid_range_invalid_to_na(column: str, invalid_value: float) -> None:
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "hours_weekly": [20, 50, 70],
            "earnings_hourly": [40, 70, 90],
        }
    ).astype("Float32")

    df.loc[1, column] = invalid_value
    exp = df.copy()
    exp.loc[1, column] = pd.NA
    got = _check_valid_range(df)
    assert_frame_equal(got, exp)


def test_filter_labour_force_subset_correct() -> None:
    df = pd.DataFrame(
        {
            "employment_status": [1, 2, 3, 1],
            "age": [10, 25, 30, 50],
        }
    )
    exp = pd.DataFrame(
        {
            "employment_status": [2, 1],
            "age": [25, 50],
        },
        index=[1, 3],
    )
    got = _filter_labour_force(df)
    assert_frame_equal(got, exp)


def test_clean_cps_data_cleaned_correct(
    raw_data: pd.DataFrame, info: pd.DataFrame
) -> None:
    got = clean_cps_data(raw_data, info)
    exp = pd.DataFrame(
        {
            "age": pd.Series([30], index=[1], dtype="Float32"),
            "sex": pd.Series([2], index=[1], dtype="category"),
            "education_level": pd.Series([41], index=[1], dtype="category"),
            "race": pd.Series([5], index=[1], dtype="category"),
            "hispanic": pd.Series([2], index=[1], dtype="category"),
            "employment_status": pd.Series([2], index=[1], dtype="category"),
            "full_part_time": pd.Series([5], index=[1], dtype="category"),
            "hours_weekly": pd.Series([50.0], index=[1], dtype="Float32"),
            "class_of_worker": pd.Series([2], index=[1], dtype="category"),
            "industry": pd.Series([20], index=[1], dtype="category"),
            "occupation": pd.Series([7], index=[1], dtype="category"),
            "earnings_hourly": pd.Series([15.0], index=[1], dtype="Float32"),
        }
    )
    assert_frame_equal(got, exp)


def test_clean_cps_data_raises_error_on_missing_variable(
    raw_data: pd.DataFrame, info: pd.DataFrame
) -> None:
    raw_data = raw_data.drop(columns=["prtage"])
    with pytest.raises(ValueError, match="not found in raw data"):
        clean_cps_data(raw_data, info)
