import pathlib
from pathlib import Path

import numpy as np
from brainio import brainio

import imlib
from imlib.source.source_files import source_custom_config_cellfinder
from imlib.general.exceptions import TransformationError
from imlib.general.system import safe_execute_command, SafeExecuteCommandError
from imlib.source.niftyreg_binaries import get_binary

from neuro.atlas_tools.misc import get_atlas_pixel_sizes

SOURCE_IMAGE_NAME = "downsampled.nii"
DEFAULT_CONTROL_POINT_FILE = "inverse_control_point_file.nii"
DEFAULT_OUTPUT_FILE_NAME = "roi_transformed.nii"
DEFAULT_TEMP_FILE_NAME = "ROI_TMP.nii"
PROGRAM_NAME = "reg_resample"


def save_brain(image, source_image_path, output_path):
    registration_config = source_custom_config_cellfinder()
    atlas_scale, transformation_matrix = get_transform_space_params(
        registration_config, source_image_path
    )
    brainio.to_nii(
        image.astype(np.int16),
        str(output_path),
        scale=atlas_scale,
        affine_transform=transformation_matrix,
    )

    print("finished generating roi image")


def get_transform_space_params(registration_config, destination_image):
    atlas = brainio.load_nii(str(destination_image), as_array=False)
    atlas_scale = atlas.header.get_zooms()
    atlas_pixel_sizes = get_atlas_pixel_sizes(registration_config)
    transformation_matrix = np.eye(4)
    for i, axis in enumerate(("x", "y", "z")):
        transformation_matrix[i, i] = atlas_pixel_sizes[axis]
    return atlas_scale, transformation_matrix


def get_transformation_matrix(self):
    atlas_pixel_sizes = get_atlas_pixel_sizes(self._atlas_config)
    transformation_matrix = np.eye(4)
    for i, axis in enumerate(("x", "y", "z")):
        transformation_matrix[i, i] = atlas_pixel_sizes[axis]
    self.transformation_matrix = transformation_matrix


def get_registration_cmd(
    program_path,
    floating_image_path,
    output_file_name,
    destination_image_filename,
    control_point_file,
):
    # TODO combine with amap.brain_registration
    cmd = "{} -cpp {} -flo {} -ref {} -res {}".format(
        program_path,
        control_point_file,
        floating_image_path,
        destination_image_filename,
        output_file_name,
    )
    return cmd


def transform_image_to_standard_space(
    reg_dir,
    image_to_transform_fname="downsampled.nii",
    output_fname="background_channel_reg_to_filtered_brain.nii",
):

    reg_dir = Path(reg_dir)
    image_to_transform = reg_dir / image_to_transform_fname
    image_with_desired_coordinate_space = reg_dir / "brain_filtered.nii"
    control_point_file = reg_dir / "inverse_control_point_file.nii"
    output_path = reg_dir / output_fname

    nifty_reg_binaries_folder = (
        imlib.source.niftyreg_binaries.get_niftyreg_binaries()
    )
    program_path = get_binary(nifty_reg_binaries_folder, PROGRAM_NAME)

    reg_cmd = get_registration_cmd(
        program_path,
        image_to_transform,
        output_path,
        image_with_desired_coordinate_space,
        control_point_file,
    )

    log_file_path = output_path.parent / "roi_transform_log.txt"
    error_file_path = output_path.parent / "roi_transform_error.txt"
    safely_execute_amap_registration(error_file_path, log_file_path, reg_cmd)
    print(f"Registered ROI image can be found at {output_path}")


def safely_execute_amap_registration(error_file_path, log_file_path, reg_cmd):
    print("Running ROI registration")
    try:
        safe_execute_command(reg_cmd, log_file_path, error_file_path)
    except SafeExecuteCommandError as err:
        raise TransformationError("ROI registration failed; {}".format(err))


def transform_all_channels_to_standard_space(reg_dir):
    p = pathlib.Path(reg_dir)
    all_channels = list(p.glob("downsampled_channel*"))
    if any("registered" in f.name for f in all_channels):
        return "looks like already processed... skipping..."

    for channel in all_channels:
        transform_image_to_standard_space(
            reg_dir,
            image_to_transform_fname=channel.name,
            output_fname=f"registered_{channel.name}",
        )


def transform_background_channel_to_standard_space(reg_dir):
    p = pathlib.Path(reg_dir)
    all_channels = list(p.glob("downsampled.nii"))
    if any("registered" in f.name for f in all_channels):
        return "looks like already processed... skipping..."

    for channel in all_channels:
        transform_image_to_standard_space(
            reg_dir,
            image_to_transform_fname=channel.name,
            output_fname=f"registered_{channel.name}",
        )
