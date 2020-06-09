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


class General(QWidget):
    def __init__(self, viewer, point_size, spline_size, *args, **kwargs):
        super(General, self).__init__(*args, **kwargs)
        self.point_size = point_size
        self.spline_size = spline_size

        self.napari_point_size = int(
            BRAINRENDER_TO_NAPARI_SCALE * self.point_size
        )
        self.napari_spline_size = int(
            BRAINRENDER_TO_NAPARI_SCALE * self.spline_size
        )

        self.x_scaling = 10
        self.y_scaling = 10
        self.z_scaling = 10
        self.track_file_extension = ".h5"
        self.image_file_extension = ".nii"
        self.spline_points = 1000
        self.spline_smoothing = 0.1
        self.summarise_track = True
        self.add_surface_point = False
        self.fit_degree = 3

        self.viewer = viewer
        self.track_layers = []
        self.label_layers = []
        self.scene = Scene(add_root=True)
        self.splines = None

        self.instantiated = False
        layout = QGridLayout()
        self.load_button = QPushButton("Load project", self)
        self.load_atlas_button = QPushButton("Load atlas", self)
        self.load_atlas_button.setVisible(False)
        self.save_button = QPushButton("Save", self)
        self.save_button.setVisible(False)
        self.status_label = QLabel()
        self.status_label.setText(f"Ready")

        layout.addWidget(self.load_button, 0, 0)
        layout.addWidget(self.load_atlas_button, 0, 1)
        layout.addWidget(self.save_button, 0, 2)

        layout.addWidget(self.status_label, 7, 1, 1, 2)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.load_button.clicked.connect(self.load_amap_directory)
        self.load_atlas_button.clicked.connect(self.load_atlas)
        self.save_button.clicked.connect(self.save)

        self.add_track_panel(layout)
        self.setLayout(layout)

        self.filename = None
        self.region_labels = []

    def add_track_panel(self, layout):
        add_track_button = QPushButton("Add track", self)
        trace_track_button = QPushButton("Trace tracks", self)

        self.summarise_checkbox = QCheckBox()
        self.summarise_checkbox.setChecked(True)

        self.track_panel = QGroupBox("Track tracing")
        track_layout = QGridLayout()
        track_layout.addWidget(QLabel("Summarise"), 0, 0)
        track_layout.addWidget(self.summarise_checkbox, 0, 1)
        track_layout.addWidget(add_track_button, 1, 0)
        track_layout.addWidget(trace_track_button, 1, 1)

        track_layout.setColumnMinimumWidth(1, 150)
        self.track_panel.setLayout(track_layout)
        layout.addWidget(self.track_panel, 1, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        add_track_button.clicked.connect(self.add_new_track)
        trace_track_button.clicked.connect(self.run_track_analysis)
        self.track_panel.setVisible(False)

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

    def load_amap_directory(self):
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

        self.load_atlas_button.setVisible(True)
        self.save_button.setVisible(True)
        self.initialise_region_segmentation()
        self.initialise_track_tracing()
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

    def initialise_region_segmentation(self):
        label_files = glob(
            str(self.paths.regions_directory)
            + "/*"
            + self.image_file_extension
        )
        if self.paths.regions_directory.exists() and label_files != []:
            for label_file in label_files:
                self.label_layers.append(
                    add_existing_label_layers(
                        self.viewer, label_file, memory=memory
                    )
                )

    def initialise_track_tracing(self):
        track_files = glob(
            str(self.paths.tracks_directory) + "/*" + self.track_file_extension
        )
        if self.paths.tracks_directory.exists() and track_files != []:
            for track_file in track_files:
                self.track_layers.append(
                    add_existing_track_layers(
                        self.viewer,
                        track_file,
                        self.napari_point_size,
                        self.x_scaling,
                        self.y_scaling,
                        self.z_scaling,
                    )
                )
        self.track_panel.setVisible(True)

    def add_new_track(self):
        print("Adding a new track\n")
        num = len(self.track_layers)
        new_track_layers = self.viewer.add_points(
            n_dimensional=True,
            size=self.napari_point_size,
            name=f"track_{num}",
        )
        new_track_layers.mode = "ADD"
        self.track_layers.append(new_track_layers)

    def run_track_analysis(self):
        print("Running track analysis")

        # global splines
        # global scene
        self.scene, self.splines = track_analysis(
            self.viewer,
            self.base_layer,
            self.scene,
            self.paths.tracks_directory,
            self.x_scaling,
            self.y_scaling,
            self.z_scaling,
            self.napari_spline_size,
            add_surface_to_points=self.add_surface_point,
            spline_points=self.spline_points,
            fit_degree=self.fit_degree,
            spline_smoothing=self.spline_smoothing,
            point_size=self.point_size,
            spline_size=self.spline_size,
            summarise_track=self.summarise_track,
            track_file_extension=self.track_file_extension,
        )
        print("Finished!\n")

    def save(self):
        print("Saving")
        worker = save_all(
            self.viewer,
            self.paths.regions_directory,
            self.paths.tracks_directory,
            self.label_layers,
            self.track_layers,
            self.paths.downsampled_image,
            self.x_scaling,
            self.y_scaling,
            self.z_scaling,
            track_file_extension=self.track_file_extension,
        )
        worker.start()
