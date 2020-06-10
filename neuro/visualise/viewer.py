import napari
from neuro.segmentation.manual_segmentation.widgets import General

from qtpy import QtCore

from pathlib import Path
from glob import glob

from brainrender.scene import Scene
from neuro.structures.IO import load_structures_as_df
from imlib.source.source_files import get_structures_path

from neuro.segmentation.paths import Paths
from neuro.generic_neuro_tools import transform_image_to_standard_space

from neuro.visualise.napari_tools.layers import (
    display_channel,
    prepare_load_nii,
)
from neuro.visualise.napari_tools.callbacks import (
    display_brain_region_name,
    region_analysis,
    track_analysis,
    save_all,
)

from neuro.segmentation.manual_segmentation.man_seg_tools import (
    add_existing_region_segmentation,
    add_existing_track_layers,
    add_new_track_layer,
    add_new_region_layer,
    view_in_brainrender,
)

import napari
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from neuro.structures.IO import load_structures_as_df

from neuro.atlas_tools.paths import Paths
from imlib.source.source_files import get_structures_path
from neuro.visualise.napari_tools.callbacks import display_brain_region_name

from neuro.visualise.napari_tools.layers import (
    display_raw,
    display_downsampled,
    display_registration,
)
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import napari
from napari.utils.io import magic_imread
from imlib.general.system import get_sorted_file_paths

from imlib.IO.cells import cells_xml_to_df, cells_to_xml
from imlib.cells.cells import Cell
from magicgui import magicgui


from neuro.gui.elements import *

from qtpy.QtWidgets import (
    QLabel,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QApplication,
    QWidget,
)


memory = False
BRAINRENDER_TO_NAPARI_SCALE = 0.3


class ViewerWidget(QWidget):
    def __init__(self, viewer):
        super(ViewerWidget, self).__init__()
        self.viewer = viewer

        self.cell_symbol = "ring"
        self.cell_opacity = 0.6
        self.cell_marker_size = 15

        self.setup_layout()

    def setup_layout(self):
        self.instantiated = False
        layout = QGridLayout()

        self.load_button = add_button(
            "Load project",
            layout,
            self.load_directory,
            0,
            0,
            minimum_width=200,
        )

        self.load_raw_data_directory_button = add_button(
            "Load data directory",
            layout,
            self.load_raw_data_directory,
            1,
            0,
            visibility=False,
        )
        self.load_raw_data_single_button = add_button(
            "Load single image",
            layout,
            self.load_raw_data_single,
            2,
            0,
            visibility=False,
        )

        self.load_registration_button = add_button(
            "Load registration",
            layout,
            self.load_registration,
            3,
            0,
            visibility=False,
        )
        self.load_cells_button = add_button(
            "Load cells", layout, self.load_cells, 4, 0, visibility=False,
        )
        self.save_cells_button = add_button(
            "Save cells", layout, self.save_cells, 5, 0, visibility=False,
        )

        self.setLayout(layout)

    def load_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dir = QFileDialog.getExistingDirectory(
            self, "Select cellfinder directory", options=options,
        )
        self.directory = Path(dir)
        self.initialise_paths()
        self.load_button.setMinimumWidth(0)
        self.load_registration_button.setVisible(True)
        self.load_raw_data_directory_button.setVisible(True)
        # self.load_raw_data_single_button.setVisible(True)
        self.load_cells_button.setVisible(True)

    def initialise_paths(self):
        self.cells = self.directory / "cells.xml"
        self.classified_cells = self.directory / "cell_classification.xml"

    def load_raw_data_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dir = QFileDialog.getExistingDirectory(
            self, "Select data directory", options=options,
        )
        dir = Path(dir)
        img_paths = get_sorted_file_paths(dir, file_extension=".tif*")
        images = magic_imread(img_paths, use_dask=True, stack=True)
        self.viewer.add_image(images, name=dir.stem)

    def load_raw_data_single(self):
        pass

    def load_registration(self):
        pass

    def load_cells(self):
        print("Loading cells")
        cells, non_cells = get_cell_arrays(str(self.classified_cells))
        self.non_cell_layer = self.viewer.add_points(
            non_cells,
            size=self.cell_marker_size,
            n_dimensional=True,
            opacity=self.cell_opacity,
            symbol=self.cell_symbol,
            face_color="lightskyblue",
            name="Non-Cells",
        )
        self.cell_layer = self.viewer.add_points(
            cells,
            size=self.cell_marker_size,
            n_dimensional=True,
            opacity=self.cell_opacity,
            symbol=self.cell_symbol,
            face_color="lightgoldenrodyellow",
            name="Cells",
        )
        self.save_cells_button.setVisible(True)
        print("Finished!")

    def save_cells(self):
        print("Saving cells")
        napari_cells_to_xml(
            self.cell_layer.data,
            self.non_cell_layer.data,
            str(self.classified_cells),
        )
        print("Finished!")


def napari_array_to_cell_list(cell_array, type=-1):
    cell_list = []
    for row in range(0, len(cell_array)):
        cell_list.append(Cell(np.flip(cell_array[row]), type))

    return cell_list


def napari_cells_to_xml(cells, non_cells, xml_file_path):
    cell_list = napari_array_to_cell_list(cells, type=Cell.CELL)
    non_cell_list = napari_array_to_cell_list(non_cells, type=Cell.UNKNOWN)

    all_cells = cell_list + non_cell_list
    cells_to_xml(all_cells, xml_file_path)


def cells_df_as_np(cells_df, new_order=[2, 1, 0], type_column="type"):
    cells_df = cells_df.drop(columns=[type_column])
    cells = cells_df[cells_df.columns[new_order]]
    cells = cells.to_numpy()
    return cells


def get_cell_arrays(cells_file):
    df = cells_xml_to_df(cells_file)

    non_cells = df[df["type"] == Cell.UNKNOWN]
    cells = df[df["type"] == Cell.CELL]

    cells = cells_df_as_np(cells)
    non_cells = cells_df_as_np(non_cells)
    return cells, non_cells


def main():
    with napari.gui_qt():
        viewer = napari.Viewer(title="cellfinder viewer")
        viewer_widget = ViewerWidget(viewer)
        viewer.window.add_dock_widget(
            viewer_widget, name="General", area="right"
        )


if __name__ == "__main__":
    main()
