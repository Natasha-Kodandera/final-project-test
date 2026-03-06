import zipfile

from cluster_analysis.config import BLD, SRC


def task_unzip_data(
    data=SRC / "data" / "raw" / "cps_data.zip",
    produces=None,
):
    """Unzip the raw data file."""
    if produces is None:
        produces = {
            "marker": BLD / "data" / ".unzip.done",
            "cps": BLD / "data" / "cps_data.csv",
            "info": BLD / "data" / "cps_variable_info.csv",
        }

    out_path = BLD / "data"
    out_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data, "r") as zip_ref:
        zip_ref.extractall(path=out_path)

    produces["marker"].write_text("unzipped")
