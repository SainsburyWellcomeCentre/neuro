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

    print("Loading manual segmentation GUI.\n ")
    with napari.gui_qt():
        global scene
        scene = Scene(add_root=True)
        global splines
        splines = None

        viewer = napari.Viewer(title="Manual segmentation")
        general = General(viewer, point_size, spline_size)
        viewer.window.add_dock_widget(general, name="General", area="right")


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
