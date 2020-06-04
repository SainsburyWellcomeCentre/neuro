import argparse

import napari
import pathlib
import numpy as np

from PySide2.QtWidgets import QApplication
from brainio import brainio
from glob import glob

from neuro.visualise.brainrender import volume_to_vector_array_to_obj_file
from neuro.segmentation.lesion_and_track_tools.lesion_and_track_estimation import (
    get_fiber_track,
)
from neuro.generic_neuro_tools import (
    transform_background_channel_to_standard_space,
)
from neuro.visualise.napari.layers import display_channel


def run_track_viewer(
    reg_dir,
    atlas_name="annotations.nii",
    background_channel_name="registered_downsampled.nii",
    output_name="registered_track_{}.nii",
):
    reg_dir = pathlib.Path(reg_dir)
    print("Starting amap viewer")

    with napari.gui_qt():
        v = napari.Viewer(title="track viewer")
        display_channel(v, reg_dir, background_channel_name)
        points_layer = v.add_points()

        @v.bind_key("b")
        def conv_to_b(v):
            selected_points = v.layers[-1].selected_data
            if len(selected_points) == 1:
                print(f"labeled point at {v.layers[-1].data}")
                z = int(v.layers[-1].data[0][0])
                y = int(v.layers[-1].data[0][1])
                x = int(v.layers[-1].data[0][2])
                seed_point = (x, y, z)
                seed_point_str = f"{x}_{y}_{z}"
                print(f"{seed_point} extracted, getting fiber track from this")
                get_fiber_track(
                    reg_dir / atlas_name,
                    reg_dir / background_channel_name,
                    seed_point,
                    reg_dir / output_name.format(seed_point_str),
                )
                print("closing application...")
                QApplication.closeAllWindows()

        @v.bind_key("q")
        def quit(v):
            QApplication.closeAllWindows()


def load_arrays(images, names):
    with napari.gui_qt():
        v = napari.Viewer(title="track viewer")
        for image, name in zip(images, names):
            image = np.swapaxes(image, 2, 0)
            v.add_image(image, name=name)


def get_fiber_tract_in_standard_space(reg_dir):
    transform_background_channel_to_standard_space(reg_dir)
    run_track_viewer(reg_dir)
    reg_dir = pathlib.Path(reg_dir)

    segmented_files = glob(str(reg_dir) + "/registered_track*.nii")

    for segmented_file in segmented_files:
        brain = brainio.load_any(segmented_file)
        volume_to_vector_array_to_obj_file(
            brain, segmented_file.replace(".nii", ".obj")
        )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="registration_directory",
        type=str,
        help="amap/cellfinder registration output directory",
    )
    return parser


def main():
    args = get_parser().parse_args()
    get_fiber_tract_in_standard_space(args.registration_directory)


if __name__ == "__main__":
    main()
