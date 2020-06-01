import argparse
import napari

from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from PySide2.QtWidgets import QApplication
from imlib.general.system import (
    ensure_directory_exists,
    delete_directory_contents,
)

from brainio.brainio import load_any
from neuro.structures.IO import load_structures_as_df
from imlib.source.source_files import get_structures_path

from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.visualise.vis_tools import display_channel, prepare_load_nii
from neuro.visualise.brainrender import load_regions_into_brainrender
from neuro.visualise.napari import add_new_label_layer
from neuro.segmentation.manual_region_segmentation.man_seg_tools import (
    add_existing_label_layers,
    save_regions_to_file,
    analyse_region_brain_areas,
    summarise_brain_regions,
)
from neuro.segmentation.paths import Paths
import pandas as pd
from brainrender.scene import Scene
from vtkplotter.shapes import Cylinder, Line, DashedLine
import numpy as np

from brainrender.scene import Scene

import argparse
import imlib.IO.cells as cells_io
from imlib.general.numerical import check_positive_float, check_positive_int
from imlib.general.system import ensure_directory_exists
from pathlib import Path


memory = True


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
        paths.tmp__inverse_transformed_image, memory=memory
    )

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
        points_layers.append(viewer.add_points())

        # # keep for multiple probes
        # @viewer.bind_key("Control-N")
        # def add_region(viewer):
        #     """
        #     Add new region
        #     """
        #     print("\nAdding new region")
        #     label_layers.append(
        #         add_new_label_layer(
        #             viewer,
        #             registered_image,
        #             name="new_region",
        #             brush_size=brush_size,
        #             num_colors=num_colors,
        #         )
        #     )
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
            cells = viewer.layers[1].data.astype(np.int16)
            # cells = (cells * 10)
            print(cells)
            cells = pd.DataFrame(cells)

            cells.columns = ["x", "y", "z"]

            # this weird scaling is due to the ARA coordinate space
            cells["x"] = z_scaling * cells["x"]
            cells["z"] = x_scaling * cells["z"]
            cells["y"] = y_scaling * cells["y"]
            max_z = len(viewer.layers[0].data)
            cells["x"] = (z_scaling * max_z) - cells["x"]

            print(cells)
            output_filename = "/home/adam/Desktop/track.h5"
            print(f"Saving to: {output_filename}")
            cells.to_hdf(output_filename, key="df", mode="w")

            print("Finished")
            close_viewer(viewer)

            view_in_br(
                output_filename, radius=30, points_kwargs={"radius": 100}
            )


def view_in_br(track, points_kwargs={}, **kwargs):
    # from brainrender.scene.add_probe_from_sharptrack
    cells = pd.read_hdf(track)
    scene = Scene(add_root=True)

    col_by_region = points_kwargs.pop("color_by_region", True)
    color = points_kwargs.pop("color", "salmon")
    radius = points_kwargs.pop("radius", 30)
    scene.add_cells(
        cells,
        color=color,
        color_by_region=col_by_region,
        res=12,
        radius=radius,
        **points_kwargs,
    )

    r0 = np.mean(cells.values, axis=0)
    xyz = cells.values - r0
    U, S, V = np.linalg.svd(xyz)
    direction = V.T[:, 0]

    # Find intersection with brain surface
    root_mesh = scene.atlas._get_structure_mesh("root")
    p0 = direction * np.array([-1]) + r0
    p1 = (
        direction * np.array([-15000]) + r0
    )  # end point way outside of brain, on probe trajectory though
    pts = root_mesh.intersectWithLine(p0, p1)

    # Define top/bottom coordinates to render as a cylinder
    top_coord = pts[0]
    length = np.sqrt(np.sum((cells.values[-1] - top_coord) ** 2))
    bottom_coord = top_coord + direction * length

    # Render probe as a cylinder
    probe_color = kwargs.pop("color", "blackboard")
    probe_radius = kwargs.pop("radius", 15)
    probe_alpha = kwargs.pop("alpha", 1)

    probe = Cylinder(
        [top_coord, bottom_coord],
        r=probe_radius,
        alpha=probe_alpha,
        c=probe_color,
    )

    # Add to scene
    scene.add_vtkactor(probe)
    scene.add_brain_regions(["VISp"], colors="mediumseagreen", alpha=0.6)
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
        "--preview",
        dest="preview",
        action="store_true",
        help="Preview the segmented regions in brainrender",
    )
    parser.add_argument(
        "--volumes",
        dest="volumes",
        action="store_true",
        help="Calculate the volume of each brain area included in the "
        "segmented region",
    )
    parser.add_argument(
        "--summarise",
        dest="summarise",
        action="store_true",
        help="Summarise each region (centers, volumes etc.)",
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
        volumes=args.volumes,
        summarise=args.summarise,
        shading=args.shading,
        alpha=args.alpha,
        brush_size=args.brush_size,
    )


if __name__ == "__main__":
    main()
