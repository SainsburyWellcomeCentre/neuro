import argparse
import napari

import pandas as pd
import numpy as np

from pathlib import Path
from glob import glob
from vtkplotter import mesh, Spheres, Spline
from PySide2.QtWidgets import QApplication

from brainrender.scene import Scene
from brainio.brainio import load_any
from neuro.structures.IO import load_structures_as_df
from imlib.source.source_files import get_structures_path
from imlib.general.system import (
    ensure_directory_exists,
    delete_directory_contents,
)

from neuro.segmentation.paths import Paths
from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.visualise.vis_tools import display_channel, prepare_load_nii
from neuro.visualise.brainrender import load_regions_into_brainrender
from neuro.visualise.napari import add_new_label_layer
from neuro.structures.structures_tree import (
    atlas_value_to_name,
    UnknownAtlasValue,
)
from neuro.segmentation.manual_segmentation.man_seg_tools import (
    add_existing_label_layers,
    save_regions_to_file,
    analyse_region_brain_areas,
    summarise_brain_regions,
)


memory = False
BRAINRENDER_TO_NAPARI_SCALE = 0.3


def run(
    image,
    registration_directory,
    preview=False,
    volumes=False,
    summarise=False,
    num_colors=10,
    brush_size=30,
    alpha=0.8,
    shading="flat",
    add_surface_to_points=True,
    regions_to_add=[],
    probe_sites=1000,
    fit_degree=2,
    spline_smoothing=0.05,
    region_alpha=0.3,
    point_size=30,
    spline_size=10,
):
    napari_point_size = int(BRAINRENDER_TO_NAPARI_SCALE * point_size)
    napari_spline_size = int(BRAINRENDER_TO_NAPARI_SCALE * spline_size)

    paths = Paths(registration_directory, image)
    registration_directory = Path(registration_directory)

    if not paths.tmp__inverse_transformed_image.exists():
        print(
            f"The image: '{image}' has not been transformed into standard "
            f"space, and so must be transformed before segmentation."
        )
        transform_image_to_standard_space(
            registration_directory,
            image_to_transform_fname=image,
            output_fname=paths.tmp__inverse_transformed_image,
            log_file_path=paths.tmp__inverse_transform_log_path,
            error_file_path=paths.tmp__inverse_transform_error_path,
        )
    else:
        print("Registered image exists, skipping registration")

    registered_image = prepare_load_nii(
        paths.tmp__inverse_transformed_image, memory=memory
    )

    print("\nLoading manual segmentation GUI.\n ")

    # TODO: change the instructions

    print(
        "Please 'colour in' the regions you would like to segment. \n "
        "When you are done, press 'Alt-Q' to save and exit. \n If you have "
        "used the '--preview' flag, \n the region will be shown in 3D in "
        "brainrender\n for you to inspect."
    )

    with napari.gui_qt():
        viewer = napari.Viewer(title="Manual segmentation")
        display_channel(
            viewer,
            registration_directory,
            paths.tmp__inverse_transformed_image,
            memory=memory,
            name="Image in standard space",
        )
        region_labels = viewer.add_labels(
            prepare_load_nii(paths.annotations, memory=memory),
            name="Region labels",
            opacity=0.2,
            visible=False,
        )
        structures_path = get_structures_path()
        structures_df = load_structures_as_df(structures_path)

        global label_layers
        label_layers = []

        # global points_layers
        # points_layers = []
        # points_layers.append(
        #     viewer.add_points(
        #         n_dimensional=True,
        #         size=napari_point_size,
        #         name="Track label editor",
        #     )
        # )

        label_files = glob(str(paths.regions_directory) + "/*.nii")
        if paths.regions_directory.exists() and label_files != []:
            label_layers = []
            for label_file in label_files:
                label_layers.append(
                    add_existing_label_layers(
                        viewer, label_file, memory=memory
                    )
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

        @region_labels.mouse_move_callbacks.append
        def display_region_name(layer, event):
            val = layer.get_value()
            if val != 0 and val is not None:
                try:
                    region = atlas_value_to_name(val, structures_df)
                    msg = f"{region}"
                except UnknownAtlasValue:
                    msg = "Unknown region"
            else:
                msg = "No label here!"
            layer.help = msg

        @viewer.bind_key("Control-N")
        def add_region(viewer):
            """
            Add new region
            """
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
            """
            Close viewer
            """
            print("\nClosing viewer")
            QApplication.closeAllWindows()

        @viewer.bind_key("Alt-Q")
        def save_analyse_regions(viewer):
            """
            Save segmented regions and exit
            """
            ensure_directory_exists(paths.regions_directory)
            delete_directory_contents(str(paths.regions_directory))
            if volumes:
                print("Calculating region volume distribution")
                annotations = load_any(paths.annotations)
                hemispheres = load_any(paths.hemispheres)
                structures_reference_df = load_structures_as_df(
                    get_structures_path()
                )

                print(
                    f"\nSaving summary volumes to: {paths.regions_directory}"
                )
                for label_layer in label_layers:
                    analyse_region_brain_areas(
                        label_layer,
                        paths.regions_directory,
                        annotations,
                        hemispheres,
                        structures_reference_df,
                    )
            if summarise:
                print("Summarising regions")
                summarise_brain_regions(label_layers, paths.region_summary_csv)

            print(f"\nSaving regions to: {paths.regions_directory}")
            for label_layer in label_layers:
                save_regions_to_file(
                    label_layer,
                    paths.regions_directory,
                    paths.downsampled_image,
                )
            close_viewer(viewer)

    obj_files = glob(str(paths.regions_directory) + "/*.obj")
    if obj_files:
        if preview:
            print("\nPreviewing in brainrender")
            load_regions_into_brainrender(
                obj_files, alpha=alpha, shading=shading
            )
    else:
        print("\n'--preview' selected, but no regions to display")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    main_parser = parser.add_argument_group("General options")
    main_parser.add_argument(
        dest="image",
        type=str,
        help="downsampled image to be segmented " "(as a string)",
    )
    main_parser.add_argument(
        dest="registration_directory",
        type=str,
        help="amap/cellfinder registration output directory",
    )
    main_parser.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="Preview the segmented regions in brainrender",
    )

    region_parser = parser.add_argument_group("Region segmentation options")
    region_parser.add_argument(
        "--volumes",
        dest="volumes",
        action="store_true",
        help="Calculate the volume of each brain area included in the "
        "segmented region",
    )
    region_parser.add_argument(
        "--summarise",
        dest="summarise",
        action="store_true",
        help="Summarise each region (centers, volumes etc.)",
    )
    region_parser.add_argument(
        "--shading",
        type=str,
        default="flat",
        help="Object shading type for brainrender ('flat', 'giroud' or "
        "'phong').",
    )
    region_parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Object transparency for brainrender.",
    )
    region_parser.add_argument(
        "--brush-size",
        dest="brush_size",
        type=int,
        default=30,
        help="Default size of the label brush.",
    )

    path_parser = parser.add_argument_group("Path segmentation options")
    path_parser.add_argument(
        "--surface-point",
        dest="add_surface_to_points",
        action="store_true",
        help="Find the closest part of the brain surface to the first point,"
        "and include that as a point for the spline fit.",
    )
    path_parser.add_argument(
        "--probe-sites",
        dest="probe_sites",
        type=int,
        default=1000,
        help="How many segments should the probehave",
    )
    path_parser.add_argument(
        "--fit-degree",
        dest="fit_degree",
        type=int,
        default=2,
        help="Degree of the spline fit (1<degree<5)",
    )
    path_parser.add_argument(
        "--fit-smooth",
        dest="fit_smooth",
        type=float,
        default=0.05,
        help="Smoothing factor for the spline fit, between 0 (interpolate "
        "points exactly) and 1 (average point positions).",
    )
    parser.add_argument(
        "--spline-radius",
        dest="spline_size",
        type=int,
        default=10,
        help="Radius of the visualised spline",
    )
    path_parser.add_argument(
        "--point-radius",
        dest="point_size",
        type=int,
        default=30,
        help="Radius of the visualised points",
    )
    path_parser.add_argument(
        "--region-alpha",
        dest="region_alpha",
        type=float,
        default=0.4,
        help="Brain region transparency for brainrender.",
    )
    path_parser.add_argument(
        "--regions",
        dest="regions",
        default=[],
        nargs="+",
        help="Brain regions to render, as acronyms. e.g. 'VISp MOp1'",
    )

    return parser


def main():
    args = get_parser().parse_args()
    run(
        args.image,
        args.registration_directory,
        preview=args.preview,
        volumes=args.volumes,
        summarise=args.summarise,
        shading=args.shading,
        alpha=args.alpha,
        brush_size=args.brush_size,
        add_surface_to_points=args.add_surface_to_points,
        regions_to_add=args.regions,
        probe_sites=args.probe_sites,
        fit_degree=args.fit_degree,
        spline_smoothing=args.fit_smooth,
        region_alpha=args.region_alpha,
        point_size=args.point_size,
        spline_size=args.spline_size,
    )


if __name__ == "__main__":
    main()
