import napari


from pathlib import Path
from glob import glob
from PySide2.QtWidgets import QApplication
from qtpy.QtWidgets import QDoubleSpinBox, QSpinBox
from magicgui import magicgui

from brainrender.scene import Scene
from neuro.structures.IO import load_structures_as_df
from imlib.source.source_files import get_structures_path

from neuro.segmentation.paths import Paths
from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.segmentation.manual_segmentation.parser import get_parser

from neuro.visualise.napari.layers import (
    display_channel,
    prepare_load_nii,
    view_spline,
    add_new_label_layer,
)
from neuro.visualise.napari.callbacks import (
    display_brain_region_name,
    region_analysis,
    track_analysis,
)

from neuro.segmentation.manual_segmentation.man_seg_tools import (
    add_existing_label_layers,
    view_in_brainrender,
)


memory = False
BRAINRENDER_TO_NAPARI_SCALE = 0.3


def run(
    image,
    registration_directory,
    num_colors=10,
    brush_size=30,
    regions_to_add=[],
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
    with napari.gui_qt():
        global scene
        scene = Scene(add_root=True)
        global spline
        spline = None

        viewer = napari.Viewer(title="Manual segmentation")
        base_layer = display_channel(
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
        structures_df = load_structures_as_df(get_structures_path())

        global label_layers
        label_layers = []

        global points_layers
        points_layers = []

        label_files = glob(str(paths.regions_directory) + "/*.nii")
        if paths.regions_directory.exists() and label_files != []:
            label_layers = []
            for label_file in label_files:
                label_layers.append(
                    add_existing_label_layers(
                        viewer, label_file, memory=memory
                    )
                )

        @magicgui(
            call_button="Extract track",
            layout="form",
            fit_degree={"widget_type": QSpinBox, "minimum": 1, "maximum": 5},
            spline_points={
                "widget_type": QSpinBox,
                "minimum": 1,
                "maximum": 10000,
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
            spline_points: int = 1000,
            spline_smoothing: float = 0.1,
            x_scaling: float = 10,
            y_scaling: float = 10,
            z_scaling: float = 10,
            summarise_track=True,
            add_surface_point=False,
        ):
            print("Running track analysis")

            global spline
            global scene
            scene, spline = track_analysis(
                viewer,
                scene,
                paths.track_points_file,
                points_layers,
                x_scaling,
                y_scaling,
                z_scaling,
                summary_csv_file=paths.probe_summary_csv,
                add_surface_to_points=add_surface_point,
                spline_points=spline_points,
                fit_degree=fit_degree,
                spline_smoothing=spline_smoothing,
                point_size=point_size,
                spline_size=spline_size,
                summarise_track=summarise_track,
            )

            view_spline(
                viewer,
                base_layer,
                spline,
                x_scaling,
                y_scaling,
                z_scaling,
                napari_spline_size,
            )

        @magicgui(call_button="Extract regions", layout="vertical")
        def run_region_analysis(
            calculate_volumes=False, summarise_volumes=True
        ):
            print("Running region analysis!")
            region_analysis(
                label_layers,
                structures_df,
                paths.regions_directory,
                paths.annotations,
                paths.hemispheres,
                image_like=paths.downsampled_image,
                output_csv_file=paths.region_summary_csv,
                volumes=calculate_volumes,
                summarise=summarise_volumes,
            )
            QApplication.closeAllWindows()

        @magicgui(call_button="Add region")
        def new_region():
            print("Adding a new region!")
            new_label_layer = add_new_label_layer(
                viewer,
                registered_image,
                name="new_region",
                brush_size=brush_size,
                num_colors=num_colors,
            )
            new_label_layer.mode = "PAINT"
            label_layers.append(new_label_layer)

        @magicgui(call_button="Add track")
        def new_track():
            print("Adding a new track!")
            new_points_layer = viewer.add_points(
                n_dimensional=True, size=napari_point_size, name="new_track",
            )
            new_points_layer.mode = "ADD"
            points_layers.append(new_points_layer)

        @magicgui(
            call_button="View in brainrender",
            layout="vertical",
            region_alpha={
                "widget_type": QDoubleSpinBox,
                "minimum": 0,
                "maximum": 1,
                "SingleStep": 0.1,
            },
            structure_alpha={
                "widget_type": QDoubleSpinBox,
                "minimum": 0,
                "maximum": 1,
                "SingleStep": 0.1,
            },
            shading={"choices": ["flat", "giroud", "phong"]},
        )
        def to_brainrender(
            region_alpha: float = 0.8,
            structure_alpha: float = 0.8,
            shading="flat",
        ):
            print("\nClosing viewer and viewing in brainrender")
            QApplication.closeAllWindows()
            view_in_brainrender(
                scene,
                spline,
                paths.regions_directory,
                alpha=region_alpha,
                shading=shading,
                regions_to_add=regions_to_add,
                region_alpha=structure_alpha,
            )

        viewer.window.add_dock_widget(
            new_track.Gui(), name="Add track", area="right"
        )
        viewer.window.add_dock_widget(
            run_track_analysis.Gui(), name="Track analysis", area="right"
        )
        viewer.window.add_dock_widget(
            new_region.Gui(), name="Add region", area="right"
        )
        viewer.window.add_dock_widget(
            run_region_analysis.Gui(), name="Region analysis", area="right"
        )

        viewer.window.add_dock_widget(
            to_brainrender.Gui(), name="Brainrender", area="right"
        )

        @region_labels.mouse_move_callbacks.append
        def display_region_name(layer, event):
            display_brain_region_name(layer, structures_df)


def main():
    args = get_parser().parse_args()
    run(
        args.image,
        args.registration_directory,
        brush_size=args.brush_size,
        regions_to_add=args.regions,
        point_size=args.point_size,
        spline_size=args.spline_size,
    )


if __name__ == "__main__":
    main()
