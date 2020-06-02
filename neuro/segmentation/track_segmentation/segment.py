import napari

from PySide2.QtWidgets import QApplication
from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.visualise.vis_tools import display_channel, prepare_load_nii
from neuro.segmentation.paths import Paths
import pandas as pd
from vtkplotter import mesh, Spheres, Spline
import numpy as np
from brainrender.scene import Scene
import argparse
from pathlib import Path
from imlib.general.system import ensure_directory_exists

memory = True


def run(
    image,
    registration_directory,
    visualise=True,
    add_surface_to_points=True,
    regions_to_add=["VISp", "MOp1"],
    probe_sites=1000,
    fit_degree=2,
    spline_smoothing=0.05,
    region_alpha=0.3,
    point_radius=30,
    spline_radius=10,
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

    print("\nLoading probe segmentation GUI.\n ")
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
        )
        global points_layers
        points_layers = []
        points_layers.append(viewer.add_points(n_dimensional=True))

        @viewer.bind_key("Control-X")
        def close_viewer(viewer):
            """
            Close viewer
            """
            print("\nClosing viewer")
            QApplication.closeAllWindows()

        @viewer.bind_key("Alt-Q")
        def save_analyse_regions(
            viewer, x_scaling=10, y_scaling=10, z_scaling=10
        ):
            """
            Save segmented probes and exit
            """
            ensure_directory_exists(paths.tracks_directory)

            cells = viewer.layers[1].data.astype(np.int16)
            # cells = (cells * 10)
            cells = pd.DataFrame(cells)

            cells.columns = ["x", "y", "z"]

            # this weird scaling is due to the ARA coordinate space
            cells["x"] = z_scaling * cells["x"]
            cells["z"] = x_scaling * cells["z"]
            cells["y"] = y_scaling * cells["y"]
            max_z = len(viewer.layers[0].data)
            cells["x"] = (z_scaling * max_z) - cells["x"]

            print(f"Saving to: {paths.track_points_file}")
            cells.to_hdf(paths.track_points_file, key="df", mode="w")

            close_viewer(viewer)

            print("Analysing track")
            analyse_track(
                paths.track_points_file,
                summary_csv_file=paths.probe_summary_csv,
                add_surface_to_points=add_surface_to_points,
                regions_to_add=regions_to_add,
                probe_sites=probe_sites,
                fit_degree=fit_degree,
                spline_smoothing=spline_smoothing,
                region_alpha=region_alpha,
                point_radius=point_radius,
                visualise=visualise,
                spline_radius=spline_radius,
            )


def analyse_track(
    track,
    summary_csv_file=None,
    visualise=False,
    add_surface_to_points=True,
    probe_sites=1000,
    fit_degree=2,
    spline_smoothing=0.05,
    regions_to_add=[],
    region_alpha=0.3,
    point_radius=30,
    spline_radius=10,
):

    cells = pd.read_hdf(track)
    scene = Scene(add_root=True)

    points = np.array(cells)

    if add_surface_to_points:
        print(
            "Finding the closest point on the brain surface to the first point"
        )
        root_mesh = mesh.Mesh(scene.root)
        surface_intersection = np.expand_dims(
            root_mesh.closestPoint(points[0]), axis=0
        )
        points = np.concatenate([surface_intersection, points], axis=0)
        scene.add_vtkactor(
            Spheres(surface_intersection, r=point_radius).color("n")
        )
    far_point = np.expand_dims(points[-1], axis=0)

    print(
        f"Fitting a spline with {probe_sites} segments, of degree "
        f"'{fit_degree}' to the points"
    )
    spline = (
        Spline(
            points, smooth=spline_smoothing, degree=fit_degree, res=probe_sites
        )
        .pointSize(spline_radius)
        .color("n")
    )

    if summary_csv_file is not None:
        print("Determining the brain region for each segment of the spline")
        spline_regions = [
            scene.atlas.get_structure_from_coordinates(p, just_acronym=False)
            for p in spline.points().tolist()
        ]
        print(f"Saving results to: {summary_csv_file}")
        df = pd.DataFrame(
            columns=["Position", "Region ID", "Region acronym", "Region name"]
        )
        for idx, spline_region in enumerate(spline_regions):
            if spline_region is None:
                df = df.append(
                    {
                        "Position": idx,
                        "Region ID": "Not found in brain",
                        "Region acronym": "Not found in brain",
                        "Region name": "Not found in brain",
                    },
                    ignore_index=True,
                )
            else:
                df = df.append(
                    {
                        "Position": idx,
                        "Region ID": spline_region["id"],
                        "Region acronym": spline_region["acronym"],
                        "Region name": spline_region["name"],
                    },
                    ignore_index=True,
                )
        df.to_csv(summary_csv_file, index=False)
    if visualise:
        print("Visualising 3D data in brainrender")
        scene.add_cells(
            cells,
            color_by_region=True,
            res=12,
            radius=point_radius,
            verbose=False,
        )

        scene.add_vtkactor(Spheres(far_point, r=point_radius).color("n"))
        scene.add_vtkactor(spline)
        scene.add_brain_regions(regions_to_add, alpha=region_alpha)
        scene.verbose = False
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
        "--no-preview",
        dest="no_preview",
        action="store_true",
        help="Don't preview the segmented regions in brainrender",
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
        dest="spline_radius",
        type=int,
        default=10,
        help="Radius of the visualised spline",
    )
    parser.add_argument(
        "--point-radius",
        dest="point_radius",
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
        visualise=not (args.no_preview),
        add_surface_to_points=args.add_surface_to_points,
        regions_to_add=args.regions,
        probe_sites=args.probe_sites,
        fit_degree=args.fit_degree,
        spline_smoothing=args.fit_smooth,
        region_alpha=args.region_alpha,
        point_radius=args.point_radius,
        spline_radius=args.spline_radius,
    )


if __name__ == "__main__":
    main()
