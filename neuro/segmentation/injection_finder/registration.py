import os
import logging

from brainio import brainio
from imlib.register.niftyreg.transform import run_transform
from neuro.atlas_tools.paths import INVERSE_CONTROL_POINT
from neuro.atlas_tools.paths import BRAIN_FILTERED as default_atlas_name


def get_registered_image(nii_path, registration_dir, overwrite=False):
    # get file paths
    basedir = os.path.split(nii_path)[0]
    output_filename = os.path.join(
        basedir,
        "{}_transformed.nii".format(os.path.split(nii_path)[1].split(".")[0]),
    )
    if os.path.isfile(output_filename) and not overwrite:
        logging.info("Skipping registration as output file already exists")
    else:

        destination_image = os.path.join(registration_dir, default_atlas_name)
        control_point_file = os.path.join(
            registration_dir, INVERSE_CONTROL_POINT
        )

        log_file_path = os.path.join(basedir, "registration_log.txt")
        error_file_path = os.path.join(basedir, "registration_err.txt")

        logging.info("Running registration")
        run_transform(
            nii_path,
            output_filename,
            destination_image,
            control_point_file,
            log_file_path,
            error_file_path,
        )

    return brainio.load_any(output_filename)
