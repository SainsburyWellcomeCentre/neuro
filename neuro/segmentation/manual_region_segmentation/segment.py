import argparse
import napari

from pathlib import Path
from glob import glob

from PySide2.QtWidgets import QApplication
from imlib.general.system import (
    delete_temp,
    ensure_directory_exists,
    delete_directory_contents,
)

from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.visualise.vis_tools import display_channel, prepare_load_nii
from neuro.visualise.brainrender import load_regions_into_brainrender
from neuro.visualise.napari import add_new_label_layer
from neuro.segmentation.manual_region_segmentation.man_seg_tools import (
    add_existing_label_layers,
    save_regions_to_file,
)


class Paths:
    """
    A single class to hold all file paths that may be used. Any paths
    prefixed with "tmp__" refer to internal intermediate steps, and will be
    deleted if "--debug" is not used.
    """

    def __init__(self, registration_output_folder, downsampled_image):
        self.registration_output_folder = Path(registration_output_folder)
        self.downsampled_image = self.join(downsampled_image)

        self.regions_directory = self.join("segmented_regions")

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
    preview=False,
    debug=False,
    num_colors=10,
    brush_size=30,
    alpha=0.8,
    shading="flat",
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

    registered_image = prepare_load_nii(paths.tmp__inverse_transformed_image)

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

        global label_layers
        label_layers = []

        label_files = glob(str(paths.regions_directory) + "/*.nii")
        if paths.regions_directory.exists() and label_files != []:
            label_layers = []
            for label_file in label_files:
                label_layers.append(
                    add_existing_label_layers(viewer, label_file)
                )
        else:
            label_layers.append(
                add_new_label_layer(
                    viewer,
                    registered_image,
                    brush_size=brush_size,
                    num_colors=num_colors,
                )
            )

        @viewer.bind_key("Control-N")
        def add_region(viewer):
            print("\nAdding new region")
            label_layers.append(
                add_new_label_layer(
                    viewer,
                    registered_image,
                    name="new_region",
                    brush_size=brush_size,
                    num_colors=num_colors,
                )
            )

        @viewer.bind_key("Control-X")
        def close_viewer(viewer):
            print("\nClosing viewer")
            QApplication.closeAllWindows()

        @viewer.bind_key("Control-S")
        def save_regions(viewer):
            print(f"\nSaving regions to: {paths.regions_directory}")
            ensure_directory_exists(paths.regions_directory)
            delete_directory_contents(str(paths.regions_directory))

            for label_layer in label_layers:
                save_regions_to_file(
                    label_layer,
                    paths.regions_directory,
                    paths.downsampled_image,
                )
            close_viewer(viewer)

    if not debug:
        print("Deleting temporary files")
        delete_temp(paths.registration_output_folder, paths)

    if preview:
        print("\nPreviewing in brainrender")
        obj_files = glob(str(paths.regions_directory) + "/*.obj")
        load_regions_into_brainrender(obj_files, alpha=alpha, shading=shading)


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
    parser.add_argument(
        "--shading",
        type=str,
        default="flat",
        help="Object shading type for brainrender ('flat', 'giroud' or "
        "'phong').",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Object transparency for brainrender.",
    )
    parser.add_argument(
        "--brush-size",
        dest="brush_size",
        type=int,
        default=30,
        help="Default size of the label brush.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    run(
        args.image,
        args.registration_directory,
        preview=args.preview,
        debug=args.debug,
        shading=args.shading,
        alpha=args.alpha,
        brush_size=args.brush_size,
    )


if __name__ == "__main__":
    main()
