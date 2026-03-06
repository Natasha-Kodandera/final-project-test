import zipfile

from cluster_analysis.config import BLD, SRC

_PRODUCES = {
    "marker": BLD / "data" / ".unzip_done",
    "data": BLD / "data" / "cps_jan26.csv",
    "variable_info": BLD / "data" / "cps_variable_info.csv",
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
