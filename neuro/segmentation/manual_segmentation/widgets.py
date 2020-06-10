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
    def __init__(
        self,
        viewer,
        point_size=30,
        spline_size=10,
        track_file_extension=".h5",
        image_file_extension=".nii",
        x_scaling=10,
        y_scaling=10,
        z_scaling=10,
        num_colors=10,
        brush_size=30,
        spline_points_default=1000,
        spline_smoothing_default=0.1,
        fit_degree_default=3,
        summarise_track_default=True,
        add_surface_point_default=False,
        calculate_volumes_default=True,
        summarise_volumes_default=True,
        region_alpha_default=0.8,
        structure_alpha_default=0.8,
        shading_default="flat",
        region_to_add_default="",
    ):
        super(General, self).__init__()
        self.point_size = point_size
        self.spline_size = spline_size

        # general variables
        self.viewer = viewer

        self.x_scaling = x_scaling
        self.y_scaling = y_scaling
        self.z_scaling = z_scaling

        # track variables
        self.track_layers = []
        self.track_file_extension = track_file_extension
        self.spline_points_default = spline_points_default
        self.spline_smoothing_default = spline_smoothing_default
        self.summarise_track_default = summarise_track_default
        self.add_surface_point_default = add_surface_point_default
        self.fit_degree_default = fit_degree_default
        self.napari_point_size = int(
            BRAINRENDER_TO_NAPARI_SCALE * self.point_size
        )
        self.napari_spline_size = int(
            BRAINRENDER_TO_NAPARI_SCALE * self.spline_size
        )

        # region variables
        self.label_layers = []
        self.image_file_extension = image_file_extension
        self.brush_size = brush_size
        self.num_colors = num_colors
        self.calculate_volumes_default = calculate_volumes_default
        self.summarise_volumes_default = summarise_volumes_default

        # atlas variables
        self.region_labels = []

        # brainrender variables
        self.region_alpha_default = region_alpha_default
        self.structure_alpha_default = structure_alpha_default
        self.shading_default = shading_default
        self.region_to_add_default = region_to_add_default

        self.setup_layout()

    def setup_layout(self):
        self.instantiated = False
        layout = QGridLayout()

        self.load_button = add_button(
            "Load project",
            layout,
            self.load_amap_directory,
            0,
            0,
            minimum_width=200,
        )

        self.load_atlas_button = add_button(
            "Load atlas", layout, self.load_atlas, 0, 1, visibility=False
        )
        self.save_button = add_button(
            "Save", layout, self.save, 7, 1, visibility=False
        )

        self.status_label = QLabel()
        self.status_label.setText(f"Ready")

        layout.addWidget(self.status_label, 8, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.add_track_panel(layout)
        self.add_region_panel(layout)
        self.add_brainrender_panel(layout)

        self.setLayout(layout)

    def add_brainrender_panel(self, layout):
        self.initialise_brainrender()

        self.brainrender_panel = QGroupBox("brainrender")
        brainrender_layout = QGridLayout()

        add_button(
            "View in Brainrender",
            brainrender_layout,
            self.to_brainrender,
            4,
            1,
        )

        self.region_alpha = add_float_box(
            brainrender_layout,
            self.region_alpha_default,
            0,
            1,
            "Segmented region alpha",
            0.1,
            0,
        )

        self.structure_alpha = add_float_box(
            brainrender_layout,
            self.structure_alpha_default,
            0,
            1,
            "Atlas region alpha",
            0.1,
            1,
        )

        self.shading = QComboBox()
        self.shading.addItems(["flat", "giroud", "phong"])
        brainrender_layout.addWidget(QLabel("Segmented region shading"), 2, 0)
        brainrender_layout.addWidget(self.shading, 2, 1)

        self.region_to_render = QComboBox()
        self.region_to_render.addItems(self.available_meshes)
        brainrender_layout.addWidget(QLabel("Region to render"), 3, 0)
        brainrender_layout.addWidget(self.region_to_render, 3, 1)

        brainrender_layout.setColumnMinimumWidth(1, 150)
        self.brainrender_panel.setLayout(brainrender_layout)
        layout.addWidget(self.brainrender_panel, 6, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.brainrender_panel.setVisible(False)

    def add_region_panel(self, layout):
        self.region_panel = QGroupBox("Region analysis")
        region_layout = QGridLayout()

        add_button(
            "Add region", region_layout, self.add_new_region, 2, 0,
        )
        add_button(
            "Analyse regions", region_layout, self.run_region_analysis, 2, 1,
        )

        self.calculate_volumes_checkbox = add_checkbox(
            region_layout,
            self.calculate_volumes_default,
            "Calculate volumes",
            0,
        )

        self.summarise_volumes_checkbox = add_checkbox(
            region_layout,
            self.summarise_volumes_default,
            "Summarise volumes",
            1,
        )

        region_layout.setColumnMinimumWidth(1, 150)
        self.region_panel.setLayout(region_layout)
        layout.addWidget(self.region_panel, 5, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
        self.region_panel.setVisible(False)

    def add_track_panel(self, layout):
        self.track_panel = QGroupBox("Track tracing")
        track_layout = QGridLayout()

        add_button(
            "Add track", track_layout, self.add_new_track, 5, 0,
        )
        add_button(
            "Trace tracks", track_layout, self.run_track_analysis, 5, 1,
        )

        self.summarise_track_checkbox = add_checkbox(
            track_layout, self.summarise_track_default, "Summarise", 0,
        )

        self.add_surface_point_checkbox = add_checkbox(
            track_layout,
            self.add_surface_point_default,
            "Add surface point",
            1,
        )

        self.fit_degree = add_int_box(
            track_layout, self.fit_degree_default, 1, 5, "Fit degree", 2,
        )

        self.spline_smoothing = add_float_box(
            track_layout,
            self.spline_smoothing_default,
            0,
            1,
            "Spline smoothing",
            0.1,
            3,
        )

        self.spline_points = add_int_box(
            track_layout,
            self.spline_points_default,
            1,
            10000,
            "Spline points",
            4,
        )

        track_layout.setColumnMinimumWidth(1, 150)
        self.track_panel.setLayout(track_layout)
        layout.addWidget(self.track_panel, 3, 0, 1, 2)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(4)
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


def add_button(
    label,
    layout,
    connected_function,
    row,
    column,
    visibility=True,
    minimum_width=0,
):
    button = QPushButton(label)
    button.setVisible(visibility)
    button.setMinimumWidth(minimum_width)
    layout.addWidget(button, row, column)
    button.clicked.connect(connected_function)
    return button


def add_checkbox(layout, default, label, row, column=0):
    box = QCheckBox()
    box.setChecked(default)
    layout.addWidget(QLabel(label), row, column)
    layout.addWidget(box, row, column + 1)
    return box


def add_float_box(
    layout, default, minimum, maximum, label, step, row, column=0
):
    box = QDoubleSpinBox()
    box.setMinimum(minimum)
    box.setMaximum(maximum)
    box.setValue(default)
    box.setSingleStep(step)
    layout.addWidget(QLabel(label), row, column)
    layout.addWidget(box, row, column + 1)
    return box


def add_int_box(layout, default, minimum, maximum, label, row, column=0):
    box = QSpinBox()
    box.setMinimum(minimum)
    box.setMaximum(maximum)
    # Not always set if not after min & max
    box.setValue(default)
    layout.addWidget(QLabel(label), row, column)
    layout.addWidget(box, row, column + 1)
    return box
