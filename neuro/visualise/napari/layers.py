import numpy as np

from pathlib import Path
from napari.utils.io import magic_imread
from vispy.color import Colormap

from brainio import brainio
from imlib.general.system import get_sorted_file_paths

from neuro.visualise.vis_tools import (
    get_image_scales,
    get_most_recent_log,
    read_log_file,
)
from neuro.visualise.napari.utils import convert_vtk_spline_to_napari_path


label_red = Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])


def add_raw_image(viewer, image_path, name):
    """
    Add a raw image (as a virtual stack) to the napari viewer
    :param viewer: Napari viewer object
    :param image_path: Path to the raw data
    :param str name: Name to give the data
    """
    paths = get_sorted_file_paths(image_path, file_extension=".tif")
    images = magic_imread(paths, use_dask=True, stack=True)
    viewer.add_image(images, name=name, opacity=0.6, blending="additive")


def display_raw(viewer, args):
    """
    Display raw data
    :param viewer:
    :param args:
    :return:
    """
    print(
        "Starting raw image viewer. Streaming full-resolution data."
        " This may be slow."
    )
    log_entries = read_log_file(get_most_recent_log(args.amap_directory))
    config_file = Path(args.amap_directory, "config.conf")
    image_scales = get_image_scales(log_entries, config_file)
    add_raw_image(viewer, log_entries["image_paths"], name="Raw data")

    if args.raw_channels:
        for raw_image in args.raw_channels:
            name = Path(raw_image).name
            print(f"Found additional raw image to add to viewer: " f"{name}")
            add_raw_image(viewer, raw_image, name=name)

    return image_scales


def display_downsampled(viewer, args, paths):
    """
    Display downsampled data
    :param viewer:
    :param args:
    :param paths:
    :return:
    """
    image_scales = (1, 1, 1)
    load_additional_downsampled_images(
        viewer, args.amap_directory, paths, memory=args.memory
    )

    viewer.add_image(
        prepare_load_nii(paths.downsampled_brain_path, memory=args.memory),
        name="Downsampled raw data",
    )

    return image_scales


def display_registration(
    viewer, atlas, boundaries, image_scales, memory=False
):
    """
    Display results of the registration
    :param viewer: napari viewer object
    :param atlas: Annotations in sample space
    :param boundaries: Annotation boundaries in sample space
    :param tuple image_scales: Scaling of images from annotations -> data
    :param memory: Load data into memory
    """
    viewer.add_image(
        prepare_load_nii(boundaries, memory=memory),
        name="Outlines",
        contrast_limits=[0, 1],
        colormap=("label_red", label_red),
        scale=image_scales,
    )

    # labels added last so on top
    labels = viewer.add_labels(
        prepare_load_nii(atlas, memory=memory),
        name="Annotations",
        opacity=0.2,
        scale=image_scales,
    )
    return labels


def load_additional_downsampled_images(
    viewer,
    amap_directory,
    paths,
    search_string="downsampled_",
    extension=".nii",
    memory=False,
):
    """
    Loads additional downsampled (i.e. from nii) images into napari viewer
    :param viewer: Napari viewer object
    :param amap_directory: Directory containing images
    :param paths: amap paths object
    :param search_string: String that defines the images.
    Default: "downampled_"
    :param extension: File extension of the downsampled images. Default: ".nii"
    :param memory: Load data into memory
    """

    amap_directory = Path(amap_directory)

    for file in amap_directory.iterdir():
        if (
            (file.suffix == ".nii")
            and file.name.startswith(search_string)
            and file != Path(paths.downsampled_brain_path)
            and file != Path(paths.tmp__downsampled_filtered)
        ):
            print(
                f"Found additional downsampled image: {file.name}, "
                f"adding to viewer"
            )
            name = file.name.strip(search_string).strip(extension)
            viewer.add_image(
                prepare_load_nii(file, memory=memory), name=name,
            )


def prepare_load_nii(nii_path, memory=False):
    """
    Transforms a nii file into the same coordinate space as the raw data
    :param nii_path: Path to the nii file
    :param memory: Load data into memory
    :return: Numpy array in the correct coordinate space
    """
    nii_path = str(nii_path)
    image = brainio.load_any(nii_path, as_numpy=memory)
    image = np.swapaxes(image, 2, 0)
    return image


def display_channel(viewer, reg_dir, channel_fname, memory=False, name=None):
    reg_dir = Path(reg_dir)

    layer = viewer.add_image(
        prepare_load_nii(reg_dir / channel_fname, memory=memory), name=name,
    )
    return layer


def add_new_label_layer(
    viewer,
    base_image,
    name="region",
    selected_label=1,
    num_colors=10,
    brush_size=30,
):
    """
    Takes an existing napari viewer, and adds a blank label layer
    (same shape as base_image)
    :param viewer: Napari viewer instance
    :param np.array base_image: Underlying image (for the labels to be
    referencing)
    :param str name: Name of the new labels layer
    :param int selected_label: Label ID to be preselected
    :param int num_colors: How many colors (labels)
    :param int brush_size: Default size of the label brush
    :return label_layer: napari labels layer
    """
    labels = np.empty_like(base_image)
    label_layer = viewer.add_labels(labels, num_colors=num_colors, name=name)
    label_layer.selected_label = selected_label
    label_layer.brush_size = brush_size
    return label_layer


def view_spline(
    viewer,
    image_layer,
    spline,
    x_scaling,
    y_scaling,
    z_scaling,
    spline_size,
    name="Spline fit",
):
    max_z = len(image_layer.data)
    napari_spline = convert_vtk_spline_to_napari_path(
        spline, x_scaling, y_scaling, z_scaling, max_z
    )

    viewer.add_points(
        napari_spline,
        size=spline_size,
        edge_color="cyan",
        face_color="cyan",
        blending="additive",
        opacity=0.7,
        name=name,
    )
