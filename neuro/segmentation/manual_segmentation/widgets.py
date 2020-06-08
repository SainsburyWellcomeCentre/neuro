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

from neuro.visualise.napari.layers import (
    display_channel,
    prepare_load_nii,
    add_new_label_layer,
)
from neuro.visualise.napari.callbacks import (
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

import os
import enum
import heapq

import numpy as np

# import pyqtgraph as pg

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
    QAction,
)

from qtpy import QtGui
from qtpy import QtCore

# import btrack
#
# from . import utils
# from .io import TrackerFrozenState
# from .tree import _build_tree_graph
# from .layers.tracks import Tracks

import matplotlib.pyplot as plt

from typing import Union
from napari.layers import Labels
from napari._qt.qt_range_slider import QHRangeSlider

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QInputDialog,
    QLineEdit,
    QFileDialog,
)
from PyQt5.QtGui import QIcon

import napari
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

from neuro.visualise.napari.layers import (
    display_channel,
    prepare_load_nii,
    add_new_label_layer,
)
from neuro.visualise.napari.callbacks import (
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


class General(QWidget):
    def __init__(self, viewer, *args, **kwargs):
        super(General, self).__init__(*args, **kwargs)
        self.viewer = viewer
        self.instantiated = False
        layout = QGridLayout()
        self.load_button = QPushButton("Load project", self)
        self.load_atlas_button = QPushButton("Load atlas", self)
        self.load_atlas_button.setVisible(False)
        self.status_label = QLabel()
        self.status_label.setText(f"Ready")

        layout.addWidget(self.load_button, 0, 0)
        layout.addWidget(self.load_atlas_button, 0, 1)

        layout.addWidget(self.status_label, 7, 1, 1, 2)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.load_button.clicked.connect(self.load_amap_directory)
        self.load_atlas_button.clicked.connect(self.load_atlas)

        ################
        self.add_track_button = QPushButton("Add track", self)
        self.trace_track_button = QPushButton("Trace tracks", self)

        self.summarise_checkbox = QCheckBox()
        self.summarise_checkbox.setChecked(True)

        self.tracks_label = QLabel()

        # track panel
        self.track_panel = QGroupBox("Track tracing")
        track_layout = QGridLayout()
        track_layout.addWidget(QLabel("Summarise"), 0, 0)
        track_layout.addWidget(self.summarise_checkbox, 0, 1)
        track_layout.addWidget(self.add_track_button, 1, 0)
        track_layout.addWidget(self.trace_track_button, 1, 1)

        track_layout.addWidget(self.tracks_label, 4, 3)
        track_layout.setColumnMinimumWidth(1, 150)
        self.track_panel.setLayout(track_layout)
        layout.addWidget(self.track_panel, 1, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.track_panel.setVisible(False)

        self.filename = None
        self.region_labels = []

    def load_atlas(self):
        if not self.region_labels:
            self.status_label.setText(f"Loading ...")
            self.region_labels = self.viewer.add_labels(
                prepare_load_nii(self.paths.annotations, memory=memory),
                name="Region labels",
                opacity=0.2,
            )

            @self.region_labels.mouse_move_callbacks.append
            def display_region_name(layer, event):
                display_brain_region_name(layer, self.structures_df)

            self.status_label.setText(f"Ready")

    def initialise_image_view(self):
        self.set_z_position()

    def set_z_position(self):
        midpoint = int(round(len(self.base_layer.data) / 2))
        self.viewer.dims.set_point(0, midpoint)

    def load_amap_directory(
        self,
        napari_point_size=10,
        x_scaling=10,
        y_scaling=10,
        z_scaling=10,
        track_file_extension=".h5",
    ):
        self.select_nii_file()
        self.registration_directory = self.downsampled_file.parent
        self.status_label.setText(f"Loading ...")

        self.paths = Paths(self.registration_directory, self.downsampled_file)

        if not self.paths.tmp__inverse_transformed_image.exists():
            print(
                f"The image: '{self.downsampled_file}' has not been transformed into standard "
                f"space, and so must be transformed before segmentation.\n"
            )
            transform_image_to_standard_space(
                self.registration_directory,
                image_to_transform_fname=self.downsampled_file,
                output_fname=self.paths.tmp__inverse_transformed_image,
                log_file_path=self.paths.tmp__inverse_transform_log_path,
                error_file_path=self.paths.tmp__inverse_transform_error_path,
            )
        else:
            print("Registered image exists, skipping registration\n")

        registered_image = prepare_load_nii(
            self.paths.tmp__inverse_transformed_image, memory=memory
        )
        self.base_layer = display_channel(
            self.viewer,
            self.registration_directory,
            self.paths.tmp__inverse_transformed_image,
            memory=memory,
            name="Image in standard space",
        )
        self.initialise_image_view()

        self.structures_df = load_structures_as_df(get_structures_path())

        global label_layers
        label_layers = []

        global track_layers
        track_layers = []

        label_files = glob(str(self.paths.regions_directory) + "/*.nii")
        if self.paths.regions_directory.exists() and label_files != []:
            for label_file in label_files:
                label_layers.append(
                    add_existing_label_layers(
                        self.viewer, label_file, memory=memory
                    )
                )
        track_files = glob(
            str(self.paths.tracks_directory) + "/*" + track_file_extension
        )
        if self.paths.tracks_directory.exists() and track_files != []:
            for track_file in track_files:
                track_layers.append(
                    add_existing_track_layers(
                        self.viewer,
                        track_file,
                        napari_point_size,
                        x_scaling,
                        y_scaling,
                        z_scaling,
                    )
                )
        self.load_atlas_button.setVisible(True)
        self.track_panel.setVisible(True)

        self.status_label.setText(f"Ready")

    def select_nii_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select the downsampled image you wish to segment",
            "",
            "Nifti files (*.nii)",
            options=options,
        )
        self.downsampled_file = Path(file)
