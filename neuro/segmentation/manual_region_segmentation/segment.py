import argparse
import napari

from pathlib import Path
import numpy as np

from PySide2.QtWidgets import QApplication
from brainrender.scene import Scene
from imlib.general.system import delete_temp

from neuro.generic_neuro_tools import (
    transform_image_to_standard_space,
    save_brain,
)
from neuro.visualise.vis_tools import display_channel, prepare_load_nii
from neuro.brain_render_tools import volume_to_vector_array_to_obj_file


class Paths:
    """
    A single class to hold all file paths that may be used. Any paths
    prefixed with "tmp__" refer to internal intermediate steps, and will be
    deleted if "--debug" is not used.
    """

    def __init__(self, registration_output_folder, downsampled_image):
        self.registration_output_folder = Path(registration_output_folder)
        self.downsampled_image = self.join(downsampled_image)

        self.regions_object_file = self.join("regions.obj")
        self.regions_image_file = self.join("regions.nii")

        self.tmp__inverse_transformed_image = self.join(
            "image_standard_space.nii"
        )
        self.tmp__inverse_transform_log_path = self.join(
            "inverse_transform_log.txt"
        )
        self.tmp__inverse_transform_error_path = self.join(
            "inverse_transform_error.txt"
        )

    def join(self, filename):
        return self.registration_output_folder / filename


def run(
    image,
    registration_directory,
    save_segmented_image=False,
    preview=False,
    debug=False,
):
    paths = Paths(registration_directory, image)
    registration_directory = Path(registration_directory)

    if not paths.tmp__inverse_transformed_image.exists():
        transform_image_to_standard_space(
            registration_directory,
            image_to_transform_fname=image,
            output_fname=paths.tmp__inverse_transformed_image,
            log_file_path=paths.tmp__inverse_transform_log_path,
            error_file_path=paths.tmp__inverse_transform_error_path,
        )
    else:
        print("Registered image exists, skipping")

    registered_image = prepare_load_nii(
        paths.tmp__inverse_transformed_image, memory=False
    )
    labels = np.empty_like(registered_image)
    print("\nLoading manual segmentation GUI.\n ")
    print(
        "Please 'colour in' the regions you would like to segment. \n "
        "When you are done, press Ctrl+S to save and exit. \n If you have "
        "used the '--preview' flag, \n the region will be shown in 3D in "
        "brainrender\n for you to inspect."
    )

    with napari.gui_qt():
        viewer = napari.Viewer(title="Manual segmentation")
        display_channel(
            viewer,
            registration_directory,
            paths.tmp__inverse_transformed_image,
        )
        labels_layer = viewer.add_labels(labels, num_colors=20, name="Regions")

        @viewer.bind_key("Control-S")
        def add_region(viewer):
            print(f"\nSaving regions to: {paths.regions_object_file}")
            # return image back to original orientation (reoriented for napari)
            data = np.swapaxes(labels_layer.data, 2, 0)

            volume_to_vector_array_to_obj_file(data, paths.regions_object_file)
            if save_segmented_image:
                save_brain(
                    data, paths.downsampled_image, paths.regions_image_file,
                )

            print("\nClosing viewer")
            QApplication.closeAllWindows()

    if not debug:
        print("Deleting tempory files")
        delete_temp(paths.registration_output_folder, paths)

    if preview:
        print("\nPreviewing in brainrender")
        scene = Scene()
        scene.add_from_file(paths.regions_object_file, c="coral")
        scene.render()


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
    parser.add_argument(
        "--save-image",
        dest="save_image",
        action="store_true",
        help="Store the resulting segmented region image (e.g. for inspecting "
        "in 2D.",
    )
    parser.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="Preview the segmented regions in brainrender",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Debug mode. Will increase verbosity of logging and save all "
        "intermediate files for diagnosis of software issues.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    run(
        args.image,
        args.registration_directory,
        save_segmented_image=args.save_image,
        preview=args.preview,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
