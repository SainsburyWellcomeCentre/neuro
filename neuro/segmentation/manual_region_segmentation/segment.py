import argparse
import napari

from pathlib import Path
import numpy as np

from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.visualise.vis_tools import display_channel, prepare_load_nii


def run(
    image,
    registration_directory,
    temp_reg_image_name="image_standard_space.nii",
):

    registration_directory = Path(registration_directory)
    registered_image = registration_directory / temp_reg_image_name
    if not registered_image.exists():
        transform_image_to_standard_space(
            registration_directory,
            image_to_transform_fname=image,
            output_fname=temp_reg_image_name,
        )
    else:
        print("Registered image exists, skipping")

    registered_image = prepare_load_nii(registered_image, memory=False)
    labels = np.empty_like(registered_image)
    with napari.gui_qt():
        viewer = napari.Viewer(title="Manual segmentation")
        display_channel(viewer, registration_directory, temp_reg_image_name)
        labels_layer = viewer.add_labels(labels, name="Regions")

        @viewer.bind_key("Alt-A")
        def add_region(viewer):
            a = 1


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="image",
        type=str,
        help="downsampled image to be segmented " "(as a string)",
    )
    parser.add_argument(
        dest="registration_directory",
        type=str,
        help="amap/cellfinder registration output directory",
    )
    return parser


def main():
    args = get_parser().parse_args()
    run(args.image, args.registration_directory)


if __name__ == "__main__":
    main()
