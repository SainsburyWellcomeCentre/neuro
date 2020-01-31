import sys
from pathlib import Path
import pandas as pd

from neuro.points.points_to_brainrender import (
    main as points_to_brainrender_run,
)

points_dir = Path("tests", "data", "points")
cellfinder_out = points_dir / "cellfinder_out.xml"
brainrender_file = points_dir / "brainrender.h5"


def test_points_to_brainrender(tmpdir):
    tmpdir = Path(tmpdir)
    brainrender_file_test = tmpdir / "brainrender_test.h5"

    args = [
        "points_to_brainrender",
        str(cellfinder_out),
        str(brainrender_file_test),
        "-x",
        "10",
        "-y",
        "10",
        "-z",
        "10",
        "--max-z",
        "13200",
        "--hdf-key",
        "df",
    ]
    sys.argv = args
    points_to_brainrender_run()

    assert (
        (
            pd.read_hdf(brainrender_file_test, key="df")
            == pd.read_hdf(brainrender_file, key="df")
        )
        .all()
        .all()
    )
