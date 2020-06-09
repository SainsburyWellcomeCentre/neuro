import napari


from pathlib import Path
from glob import glob

from qtpy.QtWidgets import QDoubleSpinBox, QSpinBox
from magicgui import magicgui

from brainrender.scene import Scene
from neuro.structures.IO import load_structures_as_df
from imlib.source.source_files import get_structures_path

from neuro.segmentation.paths import Paths
from neuro.generic_neuro_tools import transform_image_to_standard_space
from neuro.segmentation.manual_segmentation.parser import get_parser

from neuro.visualise.napari_tools.layers import (
    display_channel,
    prepare_load_nii,
    add_new_label_layer,
)
from neuro.visualise.napari_tools.callbacks import (
    display_brain_region_name,
    region_analysis,
    track_analysis,
    save_all,
)

from neuro.segmentation.manual_segmentation.man_seg_tools import (
    add_existing_label_layers,
    add_existing_track_layers,
    view_in_brainrender,
)

from neuro.segmentation.manual_segmentation.widgets import General
from qtpy.QtWidgets import (
    QButtonGroup,
    QWidget,
    QPushButton,
    QSlider,
    QCheckBox,
    QLabel,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QComboBox,
    QGridLayout,
    QGroupBox,
)

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QInputDialog,
    QLineEdit,
    QFileDialog,
)
from PyQt5.QtGui import QIcon


memory = False
BRAINRENDER_TO_NAPARI_SCALE = 0.3


def run(
    image,
    registration_directory,
    num_colors=10,
    brush_size=30,
    point_size=30,
    spline_size=10,
    track_file_extension=".h5",
):
    global x_scaling
    global y_scaling
    global z_scaling
    x_scaling = 10
    y_scaling = 10
    z_scaling = 10

    # napari_point_size = int(BRAINRENDER_TO_NAPARI_SCALE * point_size)
    # napari_spline_size = int(BRAINRENDER_TO_NAPARI_SCALE * spline_size)

    # paths = Paths(registration_directory, image)
    # registration_directory = Path(registration_directory)

    print("Loading manual segmentation GUI.\n ")
    with napari.gui_qt():
        global scene
        scene = Scene(add_root=True)
        global splines
        splines = None

        viewer = napari.Viewer(title="Manual segmentation")
        general = General(viewer, point_size, spline_size)
        viewer.window.add_dock_widget(general, name="General", area="right")

        #
        # @magicgui(call_button="Analyse regions", layout="vertical")
        # def run_region_analysis(
        #     calculate_volumes=False, summarise_volumes=True
        # ):
        #     print("Running region analysis")
        #     worker = region_analysis(
        #         label_layers,
        #         structures_df,
        #         paths.regions_directory,
        #         paths.annotations,
        #         paths.hemispheres,
        #         output_csv_file=paths.region_summary_csv,
        #         volumes=calculate_volumes,
        #         summarise=summarise_volumes,
        #     )
        #     worker.start()
        #
        # @magicgui(call_button="Add region")
        # def new_region():
        #     print("Adding a new region\n")
        #     num = len(label_layers)
        #     new_label_layer = add_new_label_layer(
        #         viewer,
        #         registered_image,
        #         name=f"region_{num}",
        #         brush_size=brush_size,
        #         num_colors=num_colors,
        #     )
        #     new_label_layer.mode = "PAINT"
        #     label_layers.append(new_label_layer)

        #
        # available_meshes = scene.atlas.all_avaliable_meshes
        # available_meshes.append("")
        #
        # @magicgui(
        #     call_button="View in brainrender",
        #     layout="vertical",
        #     region_to_add={"choices": available_meshes},
        #     region_alpha={
        #         "widget_type": QDoubleSpinBox,
        #         "minimum": 0,
        #         "maximum": 1,
        #         "SingleStep": 0.1,
        #     },
        #     structure_alpha={
        #         "widget_type": QDoubleSpinBox,
        #         "minimum": 0,
        #         "maximum": 1,
        #         "SingleStep": 0.1,
        #     },
        #     shading={"choices": ["flat", "giroud", "phong"]},
        # )
        # def to_brainrender(
        #     region_alpha: float = 0.8,
        #     structure_alpha: float = 0.8,
        #     shading="flat",
        #     region_to_add="",
        # ):
        #     print("Closing viewer and viewing in brainrender.")
        #     QApplication.closeAllWindows()
        #     view_in_brainrender(
        #         scene,
        #         splines,
        #         paths.regions_directory,
        #         alpha=region_alpha,
        #         shading=shading,
        #         region_to_add=region_to_add,
        #         region_alpha=structure_alpha,
        #     )
        #

        # viewer.window.add_dock_widget(
        #     new_region.Gui(), name="Add region", area="left"
        # )
        # viewer.window.add_dock_widget(
        #     run_region_analysis.Gui(), name="Region analysis", area="right"
        # )
        #
        # viewer.window.add_dock_widget(
        #     to_brainrender.Gui(), name="Brainrender", area="right"
        # )


def main():
    args = get_parser().parse_args()
    run(
        args.image,
        args.registration_directory,
        brush_size=args.brush_size,
        point_size=args.point_size,
        spline_size=args.spline_size,
    )


if __name__ == "__main__":
    main()
