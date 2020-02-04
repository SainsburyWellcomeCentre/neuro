import os

from brainio import brainio

from imlib.general.system import (
    safe_execute_command,
    SafeExecuteCommandError,
)

from imlib.general.exceptions import RegistrationError
from imlib.source.niftyreg_binaries import get_binary, get_niftyreg_binaries


PROGRAM_NAME = "reg_resample"
DEFAULT_CONTROL_POINT_FILE = "inverse_control_point_file.nii"
default_atlas_name = "brain_filtered.nii"


def prepare_segmentation_cmd(
    program_path,
    floating_image_path,
    output_file_name,
    destination_image_filename,
    control_point_file,
):
    cmd = "{} -cpp {} -flo {} -ref {} -res {}".format(
        program_path,
        control_point_file,
        floating_image_path,
        destination_image_filename,
        output_file_name,
    )
    return cmd


def get_registered_image(nii_path, registration_dir, logging, overwrite=False):
    # get binaries
    nifty_reg_binaries_folder = get_niftyreg_binaries()
    program_path = get_binary(nifty_reg_binaries_folder, PROGRAM_NAME)

    # get file paths
    basedir = os.path.split(nii_path)[0]
    output_filename = os.path.join(
        basedir,
        "{}_transformed.nii".format(os.path.split(nii_path)[1].split(".")[0]),
    )
    if os.path.isfile(output_filename) and not overwrite:
        run = False
    else:
        run = True

    if run:
        destination_image = os.path.join(registration_dir, default_atlas_name)
        control_point_file = os.path.join(
            registration_dir, DEFAULT_CONTROL_POINT_FILE
        )

        log_file_path = os.path.join(basedir, "registration_log.txt")
        error_file_path = os.path.join(basedir, "registration_err.txt")

        reg_cmd = prepare_segmentation_cmd(
            program_path,
            nii_path,
            output_filename,
            destination_image,
            control_point_file,
        )
        logging.info("Running registration")
        try:
            safe_execute_command(reg_cmd, log_file_path, error_file_path)
        except SafeExecuteCommandError as err:
            raise RegistrationError("Registration failed; {}".format(err))
    else:
        logging.info("Skipping registration as output file already exists")

    return brainio.load_any(output_filename)
