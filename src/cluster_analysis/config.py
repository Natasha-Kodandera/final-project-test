"""All the general configuration of the project."""

from pathlib import Path

SRC: Path = Path(__file__).parent.resolve()
ROOT: Path = SRC.joinpath("..", "..").resolve()

BLD: Path = ROOT.joinpath("bld").resolve()


DOCUMENTS: Path = ROOT.joinpath("documents").resolve()

CONTINUOUS_VARS: tuple[str, ...] = ("age", "hours_weekly", "earnings_hourly")
CATEGORICAL_VARS: tuple[str, ...] = (
    "sex",
    "education_level",
    "race",
    "hispanic",
    "employment_status",
    "full_part_time",
    "class_of_worker",
    "industry",
    "occupation",
)

RANDOM_STATE: int = 123
