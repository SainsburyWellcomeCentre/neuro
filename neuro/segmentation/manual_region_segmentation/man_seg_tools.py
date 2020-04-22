import numpy as np
from pathlib import Path

from neuro.visualise.vis_tools import prepare_load_nii
from neuro.generic_neuro_tools import save_brain
from neuro.visualise.brainrender import volume_to_vector_array_to_obj_file


def add_existing_label_layers(
    viewer, label_file, selected_label=1, num_colors=10, brush_size=30,
):
    """
    Loads an existing (nii) image as a napari labels layer
    :param viewer: Napari viewer instance
    :param label_file: Filename of the image to be loaded
    :param int selected_label: Label ID to be preselected
    :param int num_colors: How many colors (labels)
    :param int brush_size: Default size of the label brush
    :return label_layer: napari labels layer
    """
    label_file = Path(label_file)
    labels = prepare_load_nii(label_file)
    label_layer = viewer.add_labels(
        labels, num_colors=num_colors, name=label_file.stem,
    )
    label_layer.selected_label = selected_label
    label_layer.brush_size = brush_size
    return label_layer


def save_regions_to_file(
    label_layer,
    destination_directory,
    template_image,
    obj_ext=".obj",
    image_extension=".nii",
    ignore_empty=True,
):
    """
    Saves the segmented regions to file (both as .obj and .nii)
    :param label_layer: napari labels layer (with segmented regions)
    :param destination_directory: Where to save files to
    :param template_image: Existing image of size/shape of the
    destination images
    :param obj_ext: File extension for the obj files
    :param image_extension: File extension fo the image files
    :param ignore_empty: If True, don't attempt to save empty images
    """
    data = label_layer.data
    if ignore_empty:
        if data.sum() == 0:
            return

    # swap data back to original orientation from napari orientation
    data = np.swapaxes(data, 2, 0)
    name = label_layer.name
    filename = destination_directory / (name + obj_ext)
    volume_to_vector_array_to_obj_file(
        data, filename,
    )
    filename = destination_directory / (name + image_extension)
    save_brain(
        data, template_image, filename,
    )
