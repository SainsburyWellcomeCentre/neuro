import napari
import argparse

from pathlib import Path

from imlib.source.source_files import get_structures_path

from neuro.segmentation.paths import Paths
from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.visualise.vis_tools import display_channel, prepare_load_nii
from neuro.structures.IO import load_structures_as_df
from neuro.structures.structures_tree import (
    atlas_value_to_name,
    UnknownAtlasValue,
)
from neuro.segmentation.manual_segmentation.man_seg_tools import (
    convert_vtk_spline_to_napari_path,
    analyse_track,
    analyse_track_anatomy,
    display_track_in_brainrender,
    convert_and_save_points,
)

memory = False

BRAINRENDER_TO_NAPARI_SCALE = 0.3


def run(
    image,
    registration_directory,
    preview=True,
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

    print("\nLoading probe segmentation GUI.\n ")
    print(
        "Please add points to trace the track you are interested in. \n "
        "When you are done, press 'Alt-Q' to save and exit. \n "
        "Unless you have used the '--no-preview' flag "
        "the region will be shown in 3D in "
        "brainrender for you to inspect.\n"
    )

    with napari.gui_qt():
        viewer = napari.Viewer(title="Manual segmentation")
        display_channel(
            viewer,
            registration_directory,
            paths.tmp__inverse_transformed_image,
            memory=memory,
            name="Image",
        )
        labels = viewer.add_labels(
            prepare_load_nii(paths.annotations, memory=memory),
            name="Region labels",
            opacity=0.2,
            visible=False,
        )
        structures_path = get_structures_path()
        structures_df = load_structures_as_df(structures_path)

        global points_layers
        points_layers = []
        points_layers.append(
            viewer.add_points(
                n_dimensional=True,
                size=napari_point_size,
                name="Track label editor",
            )
        )

        @labels.mouse_move_callbacks.append
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

        @viewer.bind_key("Alt-T")
        def save_analyse_regions(
            viewer, x_scaling=10, y_scaling=10, z_scaling=10
        ):
            """
            Save segmented probes and exit
            """
            max_z = len(viewer.layers[0].data)
            convert_and_save_points(
                points_layers,
                paths.track_points_file,
                x_scaling,
                y_scaling,
                z_scaling,
                max_z,
            )

            print("Analysing track")
            global scene
            global spline
            scene, spline = analyse_track(
                paths.track_points_file,
                add_surface_to_points=add_surface_to_points,
                spline_points=probe_sites,
                fit_degree=fit_degree,
                spline_smoothing=spline_smoothing,
                point_radius=point_size,
                spline_radius=spline_size,
            )
            analyse_track_anatomy(scene, spline, paths.probe_summary_csv)
            napari_spline = convert_vtk_spline_to_napari_path(
                spline, x_scaling, y_scaling, z_scaling, max_z
            )

            viewer.add_points(
                napari_spline,
                size=napari_spline_size,
                edge_color="cyan",
                face_color="cyan",
                blending="additive",
                opacity=0.7,
                name="Spline fit",
            )

    if preview:
        display_track_in_brainrender(
            scene,
            spline,
            regions_to_add=regions_to_add,
            region_alpha=region_alpha,
        )


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
        "--surface-point",
        dest="add_surface_to_points",
        action="store_true",
        help="Find the closest part of the brain surface to the first point,"
        "and include that as a point for the spline fit.",
    )
    parser.add_argument(
        "--probe-sites",
        dest="probe_sites",
        type=int,
        default=1000,
        help="How many segments should the probehave",
    )
    parser.add_argument(
        "--fit-degree",
        dest="fit_degree",
        type=int,
        default=2,
        help="Degree of the spline fit (1<degree<5)",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--point-radius",
        dest="point_size",
        type=int,
        default=30,
        help="Radius of the visualised points",
    )
    parser.add_argument(
        "--region-alpha",
        dest="region_alpha",
        type=float,
        default=0.4,
        help="Brain region transparency for brainrender.",
    )
    parser.add_argument(
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
