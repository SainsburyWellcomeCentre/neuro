import napari


from pathlib import Path
from glob import glob
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
    convert_vtk_spline_to_napari_path,
    analyse_track,
    analyse_track_anatomy,
    display_track_in_brainrender,
    convert_and_save_points,
)
from neuro.segmentation.manual_segmentation.parser import get_parser

import enum
import numpy
import napari
from napari.layers import Image
from magicgui import magicgui

from magicgui import magicgui
from magicgui._qt import QDoubleSlider
import napari
from napari.layers import Image
import skimage.data
import skimage.filters
from qtpy.QtWidgets import QSlider, QDoubleSpinBox, QSpinBox

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
    # probe_sites=1000,
    # fit_degree=2,
    # spline_smoothing=0.05,
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
        "When you are done, press 'Alt-R' to save and exit. \n If you have "
        "used the '--preview' flag, \n the region will be shown in 3D in "
        "brainrender\n for you to inspect."
    )

    with napari.gui_qt():
        global scene
        scene = Scene(add_root=True)

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

        global points_layers
        points_layers = []
        points_layers.append(
            viewer.add_points(
                n_dimensional=True,
                size=napari_point_size,
                name="Track label editor",
            )
        )

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

        @magicgui(
            call_button="Extract track",
            name={"fixedWidth": 500},
            fit_degree={"widget_type": QSpinBox, "minimum": 1, "maximum": 5},
            spline_points={
                "widget_type": QSpinBox,
                "minimum": 1,
                "maximum": 1000,
            },
            spline_smoothing={
                "widget_type": QDoubleSpinBox,
                "minimum": 0,
                "maximum": 1,
                "SingleStep": 0.1,
            },
        )
        def run_track_analysis(
            fit_degree: int = 3,
            spline_points: int = 100,
            spline_smoothing: float = 0.1,
        ):
            print(
                f"Running track analysis with fit degree: {fit_degree},"
                f"spline points: {spline_points} and spline_smoothing: "
                f"{spline_smoothing}"
            )
            x_scaling = 10
            y_scaling = 10
            z_scaling = 10

            max_z = len(viewer.layers[0].data)
            convert_and_save_points(
                points_layers,
                paths.track_points_file,
                x_scaling,
                y_scaling,
                z_scaling,
                max_z,
            )

            global spline
            global scene
            scene, spline = analyse_track(
                scene,
                paths.track_points_file,
                add_surface_to_points=add_surface_to_points,
                spline_points=spline_points,
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

        track_gui = run_track_analysis.Gui()
        viewer.window.add_dock_widget(track_gui)

        @magicgui(call_button="Extract region", layout="vertical")
        def run_region_analysis():
            print("Running region analysis!")

        region_gui = run_region_analysis.Gui()
        viewer.window.add_dock_widget(region_gui)

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

        @viewer.bind_key("Alt-R")
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

    if preview:
        obj_files = glob(str(paths.regions_directory) + "/*.obj")
        if obj_files:
            scene = load_regions_into_brainrender(
                scene, obj_files, alpha=alpha, shading=shading
            )

        scene = display_track_in_brainrender(
            scene,
            spline,
            regions_to_add=regions_to_add,
            region_alpha=region_alpha,
        )
        scene.render()


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
        # probe_sites=args.probe_sites,
        # fit_degree=args.fit_degree,
        # spline_smoothing=args.fit_smooth,
        region_alpha=args.region_alpha,
        point_size=args.point_size,
        spline_size=args.spline_size,
    )


if __name__ == "__main__":
    main()
