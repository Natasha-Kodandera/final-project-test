import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cluster_analysis.analysis.clustered_data import create_clustered_data


class DummyModel:
    def __init__(self, labels: list[int]) -> None:
        self.labels_ = labels


@pytest.fixture
def cleaned_data() -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "age": pd.Series([25, 40, 55], dtype="Float32"),
            "hours_weekly": pd.Series([20, 40, 50], dtype="Float32"),
        }
    )
    return out


def test_create_clustered_data_adds_cluster_label(
    cleaned_data: pd.DataFrame,
) -> None:
    model = DummyModel(labels=[0, 1, 1])

    got = create_clustered_data(cleaned_data=cleaned_data, model=model)
    exp = cleaned_data.copy()
    exp["cluster"] = pd.Series([0, 1, 1], index=exp.index, dtype="Int64")
    exp["cluster"] = exp["cluster"].astype(pd.CategoricalDtype())

    assert_frame_equal(got, exp)


def test_create_clustered_data_raises_type_error_on_invalid_input() -> None:
    model = DummyModel(labels=[0, 1, 1])

    with pytest.raises(TypeError, match="must be a pandas dataframe"):
        create_clustered_data(cleaned_data=[1, 2, 3], model=model)
