from pathlib import Path

import numpy as np
from brainio import brainio
from natsort import natsorted

from napari.utils.io import magic_imread

from imlib.general.config import get_config_obj
from imlib.general.system import get_sorted_file_paths, get_text_lines
from vispy.color import Colormap

label_red = Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])


def display_channel(viewer, reg_dir, channel_fname):
    """
    Display downsampled data
    :param viewer:
    :param args:
    :param paths:
    :return:
    """
    reg_dir = Path(reg_dir)

    viewer.add_image(
        prepare_load_nii(reg_dir / channel_fname, memory=False),
        name="Downsampled filtered_brain_space",
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


def get_image_scales(log_entries, config_file):
    """
    Returns the scaling from downsampled data to raw data
    :param log_entries: Entries parsed from the log file
    :param config_file: Path to the amap config file
    :return: Tuple of scaling factors
    """
    config_obj = get_config_obj(config_file)
    atlas_conf = config_obj["atlas"]
    pixel_sizes = atlas_conf["pixel_size"]
    x_scale = float(pixel_sizes["x"]) / float(log_entries["x_pixel_um"])
    y_scale = float(pixel_sizes["y"]) / float(log_entries["y_pixel_um"])
    z_scale = float(pixel_sizes["z"]) / float(log_entries["z_pixel_um"])
    return z_scale, y_scale, x_scale


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


def get_most_recent_log(directory, log_pattern="amap*.log"):
    """
    Returns the most recent amap log file (for parsing of arguments)
    :param directory:
    :param log_pattern: String pattern that defines the log
    :return: Path to the most recent log file
    """
    directory = Path(directory)
    return natsorted(directory.glob(log_pattern))[-1]


def read_log_file(
    log_file,
    log_entries_to_get=[
        "x_pixel_um",
        "y_pixel_um",
        "z_pixel_um",
        "image_paths",
        "registration_config",
    ],
    separator=": ",
):
    """
    Reads an amap log file, and returns a dict of entries corresponding to
    "log_entries_to_get"
    :param log_file: Path to the log file
    :param log_entries_to_get: List of strings corresponding to entries
    in the log file
    :param separator: Separator between the log item label and the entry.
    Default: ": "
    :return: A dict of the entries and labels
    """
    lines = get_text_lines(log_file)
    entries = {}
    for line in lines:
        for entry in log_entries_to_get:
            if line.startswith(entry):
                entries[entry] = line.strip(entry + separator)

    return entries
