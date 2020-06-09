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
    QDoubleSpinBox,
    QPushButton,
    QCheckBox,
    QLabel,
    QSpinBox,
    QFileDialog,
    QComboBox,
    QGridLayout,
    QGroupBox,
)

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QInputDialog,
    QLineEdit,
    QFileDialog,
)

memory = False
BRAINRENDER_TO_NAPARI_SCALE = 0.3


class General(QWidget):
    def __init__(self, viewer, point_size, spline_size):
        super(General, self).__init__()
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
        self.spline_points_default = 1000
        self.spline_smoothing_default = 0.1
        self.summarise_track_default = True
        self.add_surface_point_default = False
        self.fit_degree_default = 3

        self.brush_size = 30
        self.num_colors = 10

        self.calculate_volumes_default = True
        self.summarise_volumes_default = True

        self.viewer = viewer
        self.track_layers = []
        self.label_layers = []

        self.region_alpha_default = 0.8
        self.structure_alpha_default = 0.8
        self.shading_default = "flat"
        self.region_to_add_default = ""

        self.instantiated = False
        layout = QGridLayout()
        self.load_button = QPushButton("Load project", self)
        self.load_atlas_button = QPushButton("Load atlas", self)
        self.load_atlas_button.setVisible(False)
        self.save_button = QPushButton("Save", self)
        self.save_button.setVisible(False)
        self.status_label = QLabel()
        self.status_label.setText(f"Ready")

        self.load_button.setMinimumWidth(200)
        layout.addWidget(
            self.load_button, 0, 0,
        )
        layout.addWidget(self.load_atlas_button, 0, 1)
        layout.addWidget(self.save_button, 7, 1)

        layout.addWidget(self.status_label, 8, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.load_button.clicked.connect(self.load_amap_directory)
        self.load_atlas_button.clicked.connect(self.load_atlas)
        self.save_button.clicked.connect(self.save)

        self.add_track_panel(layout)
        self.add_region_panel(layout)
        self.add_brainrender_panel(layout)

        self.setLayout(layout)

        self.filename = None
        self.region_labels = []

    def add_brainrender_panel(self, layout):
        self.initialise_brainrender()

        view_brainrender_button = QPushButton("View in Brainrender", self)

        self.brainrender_panel = QGroupBox("brainrender")
        brainrender_layout = QGridLayout()

        self.region_alpha = QDoubleSpinBox()
        self.region_alpha.setValue(self.region_alpha_default)
        self.region_alpha.setMinimum(0)
        self.region_alpha.setMaximum(1)
        self.region_alpha.setSingleStep(0.1)
        brainrender_layout.addWidget(QLabel("Segmented region alpha"), 0, 0)
        brainrender_layout.addWidget(self.region_alpha, 0, 1)

        self.structure_alpha = QDoubleSpinBox()
        self.structure_alpha.setValue(self.structure_alpha_default)
        self.structure_alpha.setMinimum(0)
        self.structure_alpha.setMaximum(1)
        self.structure_alpha.setSingleStep(0.1)
        brainrender_layout.addWidget(QLabel("Atlas region alpha"), 1, 0)
        brainrender_layout.addWidget(self.structure_alpha, 1, 1)

        self.shading = QComboBox()
        self.shading.addItems(["flat", "giroud", "phong"])
        brainrender_layout.addWidget(QLabel("Segmented region shading"), 2, 0)
        brainrender_layout.addWidget(self.shading, 2, 1)

        self.region_to_render = QComboBox()
        self.region_to_render.addItems(self.available_meshes)
        brainrender_layout.addWidget(QLabel("Region to render"), 3, 0)
        brainrender_layout.addWidget(self.region_to_render, 3, 1)

        brainrender_layout.addWidget(view_brainrender_button, 4, 1)

        brainrender_layout.setColumnMinimumWidth(1, 150)
        self.brainrender_panel.setLayout(brainrender_layout)
        layout.addWidget(self.brainrender_panel, 6, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        view_brainrender_button.clicked.connect(self.to_brainrender)
        self.brainrender_panel.setVisible(False)

    def add_region_panel(self, layout):
        add_region_button = QPushButton("Add region", self)
        analyse_regions_button = QPushButton("Analyse regions", self)

        self.region_panel = QGroupBox("Region analysis")
        region_layout = QGridLayout()

        self.calculate_volumes_checkbox = QCheckBox()
        self.calculate_volumes_checkbox.setChecked(
            self.calculate_volumes_default
        )
        region_layout.addWidget(QLabel("Calculate volumes"), 0, 0)
        region_layout.addWidget(self.calculate_volumes_checkbox, 0, 1)

        self.summarise_volumes_checkbox = QCheckBox()
        self.summarise_volumes_checkbox.setChecked(
            self.summarise_volumes_default
        )
        region_layout.addWidget(QLabel("Summarise volumes"), 1, 0)
        region_layout.addWidget(self.summarise_volumes_checkbox, 1, 1)

        region_layout.addWidget(add_region_button, 2, 0)
        region_layout.addWidget(analyse_regions_button, 2, 1)

        region_layout.setColumnMinimumWidth(1, 150)
        self.region_panel.setLayout(region_layout)
        layout.addWidget(self.region_panel, 5, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        add_region_button.clicked.connect(self.add_new_region)
        analyse_regions_button.clicked.connect(self.run_region_analysis)
        self.region_panel.setVisible(False)

    def add_track_panel(self, layout):

        add_track_button = QPushButton("Add track", self)
        trace_track_button = QPushButton("Trace tracks", self)

        self.track_panel = QGroupBox("Track tracing")
        track_layout = QGridLayout()

        self.summarise_track_checkbox = QCheckBox()
        self.summarise_track_checkbox.setChecked(self.summarise_track_default)
        track_layout.addWidget(QLabel("Summarise"), 0, 0)
        track_layout.addWidget(self.summarise_track_checkbox, 0, 1)

        self.add_surface_point_checkbox = QCheckBox()
        self.add_surface_point_checkbox.setChecked(
            self.add_surface_point_default
        )
        track_layout.addWidget(QLabel("Add surface point"), 1, 0)
        track_layout.addWidget(self.add_surface_point_checkbox, 1, 1)

        self.fit_degree = QSpinBox()
        self.fit_degree.setValue(self.fit_degree_default)
        self.fit_degree.setMinimum(1)
        self.fit_degree.setMaximum(5)
        track_layout.addWidget(QLabel("Fit degree"), 2, 0)
        track_layout.addWidget(self.fit_degree, 2, 1)

        self.spline_smoothing = QDoubleSpinBox()
        self.spline_smoothing.setValue(self.spline_smoothing_default)
        self.spline_smoothing.setMinimum(0)
        self.spline_smoothing.setMaximum(1)
        self.spline_smoothing.setSingleStep(0.1)
        track_layout.addWidget(QLabel("Spline smoothing"), 3, 0)
        track_layout.addWidget(self.spline_smoothing, 3, 1)

        self.spline_points = QSpinBox()
        self.spline_points.setMinimum(1)
        self.spline_points.setMaximum(10000)
        self.spline_points.setValue(self.spline_points_default)
        track_layout.addWidget(QLabel("Spline points"), 4, 0)
        track_layout.addWidget(self.spline_points, 4, 1)

        track_layout.addWidget(add_track_button, 5, 0)
        track_layout.addWidget(trace_track_button, 5, 1)

        track_layout.setColumnMinimumWidth(1, 150)
        self.track_panel.setLayout(track_layout)
        layout.addWidget(self.track_panel, 3, 0, 1, 2)

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

        self.registered_image = prepare_load_nii(
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

        self.load_button.setMinimumWidth(0)
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

    def initialise_brainrender(self):
        self.scene = Scene(add_root=True)
        self.splines = None
        self.available_meshes = [""] + self.scene.atlas.all_avaliable_meshes

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
        self.region_panel.setVisible(True)
        self.brainrender_panel.setVisible(True)

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
        self.scene, self.splines = track_analysis(
            self.viewer,
            self.base_layer,
            self.scene,
            self.paths.tracks_directory,
            self.x_scaling,
            self.y_scaling,
            self.z_scaling,
            self.napari_spline_size,
            add_surface_to_points=self.add_surface_point_checkbox.isChecked(),
            spline_points=self.spline_points.value(),
            fit_degree=self.fit_degree.value(),
            spline_smoothing=self.spline_smoothing.value(),
            point_size=self.point_size,
            spline_size=self.spline_size,
            summarise_track=self.summarise_track_checkbox.isChecked(),
            track_file_extension=self.track_file_extension,
        )
        print("Finished!\n")

    def add_new_region(self):
        print("Adding a new region\n")
        num = len(self.label_layers)
        new_label_layer = add_new_label_layer(
            self.viewer,
            self.registered_image,
            name=f"region_{num}",
            brush_size=self.brush_size,
            num_colors=self.num_colors,
        )
        new_label_layer.mode = "PAINT"
        self.label_layers.append(new_label_layer)

    def run_region_analysis(self):
        print("Running region analysis")
        worker = region_analysis(
            self.label_layers,
            self.structures_df,
            self.paths.regions_directory,
            self.paths.annotations,
            self.paths.hemispheres,
            output_csv_file=self.paths.region_summary_csv,
            volumes=self.calculate_volumes_checkbox.isChecked(),
            summarise=self.summarise_volumes_checkbox.isChecked(),
        )
        worker.start()

    def to_brainrender(self):
        print("Closing viewer and viewing in brainrender.")
        QApplication.closeAllWindows()
        view_in_brainrender(
            self.scene,
            self.splines,
            self.paths.regions_directory,
            alpha=self.region_alpha.value(),
            shading=str(self.shading.currentText()),
            region_to_add=str(self.region_to_render.currentText()),
            region_alpha=self.structure_alpha.value(),
        )

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
