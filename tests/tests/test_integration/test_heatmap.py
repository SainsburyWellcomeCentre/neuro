import sys
from pathlib import Path
from brainio import brainio

from neuro.heatmap.heatmap import cli as heatmap_run

heatmap_dir = Path("tests", "data", "heatmap")
xml_file = heatmap_dir / "cells.xml"
raw_data = heatmap_dir / "raw"
atlas = heatmap_dir / "registered_atlas.nii"
heatmap = heatmap_dir / "heatmap.nii"


def test_heatmap(tmpdir):
    tmpdir = Path(tmpdir)
    heatmap_file_test = tmpdir / "heatmap_test.nii"

    args = [
        "heatmap",
        str(xml_file),
        str(heatmap_file_test),
        str(raw_data),
        str(atlas),
        "-x",
        "50",
        "-y",
        "50",
        "-z",
        "50",
        "--bin-size",
        "250",
        "--heatmap-smoothing",
        "250",
    ]

    sys.argv = args
    heatmap_run()

    heatmap_data = brainio.load_nii(heatmap, as_numpy=True).get_fdata()
    heatmap_test_data = brainio.load_nii(
        heatmap_file_test, as_numpy=True
    ).get_fdata()
    assert (heatmap_data == heatmap_test_data).all()
