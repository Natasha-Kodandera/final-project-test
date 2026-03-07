import zipfile
from pathlib import Path

import pandas as pd

from cluster_analysis.config import BLD, SRC
from cluster_analysis.data_management.clean_cps_data import clean_cps_data

_PRODUCES = {
    "marker": BLD / "data" / ".unzip_done",
    "data": BLD / "data" / "cps_jan26.csv",
    "info": BLD / "data" / "cps_variable_info.csv",
}


def task_unzip_data(
    data=SRC / "data" / "raw" / "cps_data.zip",
    produces=_PRODUCES,
) -> None:
    """Unzip the raw data file."""
    out_path = BLD / "data"
    out_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data, "r") as zip_ref:
        zip_ref.extractall(path=out_path)

    produces["marker"].write_text("unzipped")


def task_clean_cps_data(
    script: Path = SRC / "data_management" / "clean_cps_data.py",
    data: Path = BLD / "data" / "cps_jan26.csv",
    info: Path = BLD / "data" / "cps_variable_info.csv",
    produces: Path = BLD / "data" / "cps_cleaned.feather",
) -> None:
    """Clean the CPS January 2026 dataset."""
    raw = pd.read_csv(data)
    variable_info = pd.read_csv(info)
    df = clean_cps_data(raw, variable_info)
    df.to_feather(produces)
