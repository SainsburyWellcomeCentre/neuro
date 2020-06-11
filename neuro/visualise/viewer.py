import napari
import numpy as np

from pathlib import Path
from napari.utils.io import magic_imread
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from qtpy import QtCore
from qtpy.QtWidgets import (
    QLabel,
    QFileDialog,
    QGridLayout,
    QWidget,
)

from brainio import brainio
from imlib.source.source_files import get_structures_path
from imlib.general.system import get_sorted_file_paths
from imlib.IO.cells import cells_xml_to_df, cells_to_xml
from imlib.cells.cells import Cell

from neuro.structures.IO import load_structures_as_df
from neuro.atlas_tools.paths import Paths as registration_paths
from neuro.visualise.napari_tools.callbacks import display_brain_region_name
from neuro.visualise.napari_tools.layers import (
    prepare_load_nii,
    display_registration,
)
from neuro.visualise.vis_tools import (
    get_image_scales,
    get_most_recent_log,
    read_log_file,
)
from neuro.gui.elements import *


memory = False
BRAINRENDER_TO_NAPARI_SCALE = 0.3


class ViewerWidget(QWidget):
    def __init__(self, viewer, cell_symbol, cell_opacity, cell_marker_size):
        super(ViewerWidget, self).__init__()
        self.viewer = viewer

        self.cell_symbol = cell_symbol
        self.cell_opacity = cell_opacity
        self.cell_marker_size = cell_marker_size

        self.setup_layout()

    def setup_layout(self):
        self.instantiated = False
        layout = QGridLayout()

        self.load_cellfinder_project_button = add_button(
            "Load cellfinder project",
            layout,
            self.load_cellfinder_directory,
            0,
            0,
            minimum_width=200,
        )
        self.load_registration_project_button = add_button(
            "Load amap project",
            layout,
            self.load_registration_directory,
            1,
            0,
            minimum_width=200,
        )

        self.load_raw_data_directory_button = add_button(
            "Load data directory",
            layout,
            self.load_raw_data_directory,
            2,
            0,
            visibility=False,
        )
        self.load_raw_data_single_button = add_button(
            "Load single image",
            layout,
            self.load_raw_data_single,
            3,
            0,
            visibility=False,
        )

        self.load_downsampled_data_button = add_button(
            "Load downsampled_data",
            layout,
            self.load_downsampled_data,
            4,
            0,
            visibility=False,
        )

        self.load_registration_button = add_button(
            "Load registration",
            layout,
            self.load_registration,
            5,
            0,
            visibility=False,
        )
        self.load_cells_button = add_button(
            "Load cells", layout, self.load_cells, 6, 0, visibility=False,
        )
        self.save_cells_button = add_button(
            "Save cells", layout, self.save_cells, 7, 0, visibility=False,
        )
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.status_label = QLabel()

        self.status_label.setText("Ready")

        layout.addWidget(self.status_label, 8, 0)
        self.viewer._status = "TESTING"
        self.setLayout(layout)

    def load_cellfinder_directory(self):
        self.status_label.setText("Loading...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self, "Select cellfinder directory", options=options,
        )
        # deal with existing dialog
        if directory is not "":
            self.cellfinder_directory = Path(directory)
            self.initialise_cellfinder_paths()
            self.load_cellfinder_project_button.setVisible(False)
            self.load_registration_project_button.setVisible(False)
            self.load_raw_data_directory_button.setVisible(True)
            self.load_raw_data_single_button.setVisible(True)
            self.load_cells_button.setVisible(True)

            self.image_scales = self.get_registration_scaling()
            if self.image_scales is not None:
                self.load_registration_button.setVisible(True)
                self.load_downsampled_data_button.setVisible(True)
            else:
                print(
                    "Config files and logs could not be parsed to detect "
                    "the data scaling"
                )
        self.status_label.setText("Ready")

    def load_registration_directory(self):
        self.status_label.setText("Loading...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self, "Select registration directory", options=options,
        )
        # deal with existing dialog
        if directory is not "":
            self.registration_directory = Path(directory)
            self.initialise_registration_paths()
            self.load_cellfinder_project_button.setVisible(False)
            self.load_registration_project_button.setVisible(False)
            self.load_raw_data_directory_button.setVisible(True)
            self.load_raw_data_single_button.setVisible(True)

            self.image_scales = self.get_registration_scaling()
            if self.image_scales is not None:
                self.load_registration_button.setVisible(True)
                self.load_downsampled_data_button.setVisible(True)
            else:
                print(
                    "Config files and logs could not be parsed to detect "
                    "the data scaling"
                )
        self.status_label.setText("Ready")

    def initialise_cellfinder_paths(self):
        self.cells = self.cellfinder_directory / "cells.xml"
        self.classified_cells = (
            self.cellfinder_directory / "cell_classification.xml"
        )
        self.registration_directory = (
            self.cellfinder_directory / "registration"
        )
        self.initialise_registration_paths()

    def initialise_registration_paths(self):
        self.registration_paths = registration_paths(
            self.registration_directory
        )

    def load_raw_data_directory(self):
        self.status_label.setText("Loading...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(
            self, "Select data directory", options=options,
        )
        # deal with existing dialog
        if directory is not "":
            directory = Path(directory)
            img_paths = get_sorted_file_paths(
                directory, file_extension=".tif*"
            )
            images = magic_imread(img_paths, use_dask=True, stack=True)
            self.viewer.add_image(images, name=directory.stem)
        self.status_label.setText("Ready")

    def load_raw_data_single(self):
        self.status_label.setText("Loading...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.tif *.tiff *.nii)",
            options=options,
        )
        # deal with existing dialog
        if file is not "":
            file = Path(file)
            self.add_single_image(file)
        self.status_label.setText("Ready")

    def add_single_image(self, file):
        image = brainio.load_any(file, as_numpy=memory)
        # This should be generalised
        image = np.swapaxes(image, 2, 0)
        image = np.rot90(image, axes=(1, 2), k=3)
        self.viewer.add_image(image, name=file.stem)

    def get_registration_scaling(self):
        log_entries = []
        try:
            log_entries = read_log_file(
                get_most_recent_log(self.registration_directory)
            )
        except:
            try:
                log_entries = read_log_file(
                    get_most_recent_log(
                        self.cellfinder_directory,
                        log_pattern="cellfinder*.log",
                    )
                )
            except:
                pass
        if log_entries:
            config_file = Path(self.registration_directory, "config.conf")
            return get_image_scales(log_entries, config_file)
        else:
            return None

    def load_registration(self):
        self.status_label.setText("Loading...")
        self.region_labels = display_registration(
            self.viewer,
            self.registration_paths.registered_atlas_path,
            self.registration_paths.boundaries_file_path,
            self.image_scales,
            memory=memory,
        )
        self.structures_df = load_structures_as_df(get_structures_path())
        self.status_label.setText("Ready")

        @self.region_labels.mouse_move_callbacks.append
        def display_region_name(layer, event):
            display_brain_region_name(layer, self.structures_df)

    def load_downsampled_data(self,):
        self.status_label.setText("Loading...")
        self.load_main_downsampled_channel()
        self.load_additional_downsampled_channels()
        self.status_label.setText("Ready")

    def load_additional_downsampled_channels(
        self, search_string="downsampled_", extension=".nii"
    ):
        for file in self.registration_directory.iterdir():
            if (
                (file.suffix == extension)
                and file.name.startswith(search_string)
                and file
                != Path(self.registration_paths.downsampled_brain_path)
                and file
                != Path(self.registration_paths.tmp__downsampled_filtered)
            ):
                print(
                    f"Found additional downsampled image: {file.name}, "
                    f"adding to viewer"
                )
                name = (
                    file.name.strip(search_string).strip(extension)
                    + " (Downsampled)"
                )
                self.viewer.add_image(
                    prepare_load_nii(file, memory=memory),
                    name=name,
                    scale=self.image_scales,
                )

    def load_main_downsampled_channel(self):
        self.viewer.add_image(
            prepare_load_nii(
                self.registration_paths.downsampled_brain_path, memory=memory,
            ),
            scale=self.image_scales,
            name="Raw data (downsampled)",
        )

    def load_cells(self):
        self.status_label.setText("Loading...")
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
        self.status_label.setText("Ready")

    def save_cells(self):
        self.status_label.setText("Saving...")
        napari_cells_to_xml(
            self.cell_layer.data,
            self.non_cell_layer.data,
            str(self.classified_cells),
        )
        self.status_label.setText("Ready")


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


def parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--cell-symbol",
        dest="cell_symbol",
        type=str,
        default="ring",
        help="Marker symbol.",
    )
    parser.add_argument(
        "--cell-marker-size",
        dest="cell_marker_size",
        type=int,
        default=15,
        help="Marker size.",
    )
    parser.add_argument(
        "--cell-opacity",
        dest="cell_opacity",
        type=float,
        default=0.6,
        help="Opacity of the markers.",
    )
    return parser


def main():
    args = parser().parse_args()

    with napari.gui_qt():
        viewer = napari.Viewer(title="cellfinder viewer")
        viewer_widget = ViewerWidget(
            viewer,
            cell_opacity=args.cell_opacity,
            cell_symbol=args.cell_symbol,
            cell_marker_size=args.cell_marker_size,
        )
        viewer.window.add_dock_widget(
            viewer_widget, name="General", area="right"
        )


if __name__ == "__main__":
    main()
