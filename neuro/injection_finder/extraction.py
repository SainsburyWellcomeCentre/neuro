import os
import numpy as np
from pathlib import Path

from skimage.filters import gaussian as gaussian_filter
from skimage.filters import threshold_otsu
from skimage import measure

from brainio import brainio
from imlib.IO.surfaces import marching_cubes_to_obj
from imlib.image.orient import reorient_image

from neuro.injection_finder.registration import get_registered_image
from neuro.injection_finder.parsers import extraction_parser

import neuro as package_for_log

# For logging
import logging
from fancylog import fancylog


class Extractor:
    def __init__(
        self,
        img_filepath,
        registration_folder,
        logging,
        overwrite=False,
        gaussian_kernel=2,
        percentile_threshold=99.95,
        threshold_type="otsu",
        obj_path=None,
        overwrite_registration=False,
    ):

        """
            Extractor processes a downsampled.nii image to extract the location of the injection site.
            This is done by registering the image to the allen CCF, blurring, thresholding and finally a 
            marching cube algorithm to extract the surface of the injection site. 

            :param img_filepath: str, path to .nii file
            :param registration_folder: str, path to the registration folder [from cellfinder or amap]
            :param logging: instance of fancylog logger
            :param overwrite: bool, if False it will avoid overwriting files
            :gaussian_kernel: float, size of kernel used for smoothing
            :param percentile_threshold: float, in range [0, 1] percentile to use for thresholding
            :param threshold_type: str, either ['otsu', 'percentile'], type of threshold used
            :param obj_path: path to .obj file destination. 
            :param overwrite_registration: if false doesn't overwrite the registration step
        """

        # Get arguments
        self.img_filepath = img_filepath
        self.registration_folder = registration_folder
        self.logging = logging
        self.overwrite = overwrite
        self.gaussian_kernel = gaussian_kernel
        self.percentile_threshold = percentile_threshold
        self.threshold_type = threshold_type
        self.obj_path = obj_path
        self.overwrite_registration = overwrite_registration

        # Run first with the image oriented for brainrender
        image = self.setup()
        self.extract(image)

    def setup(self):
        if not os.path.isfile(self.img_filepath):
            raise FileNotFoundError(
                "The image path {} is not valid".format(self.img_filepath)
            )

        self.thresholded_savepath = (
            self.img_filepath.split(".")[0] + "_thresholded.nii"
        )

        # Get path to obj file and check if it existsts
        if self.obj_path is None:
            self.obj_path = self.img_filepath.split(".")[0] + ".obj"

        if os.path.isfile(self.obj_path) and not self.overwrite:
            self.logging.warning(
                "A file exists already at {}. \
                        Analysis will not run as overwrite is set disabled".format(
                    self.obj_path
                )
            )

        # Load image and register
        image = get_registered_image(
            self.img_filepath,
            self.registration_folder,
            self.logging,
            overwrite=self.overwrite_registration,
        )
        return image

    def extract(self, image, voxel_size=10):
        self.logging.info("Processing " + self.img_filepath)
        self.logging.info(
            "Gaussian filtering with kernel size: {}".format(
                self.gaussian_kernel
            )
        )

        # Gaussian filter
        kernel_shape = [self.gaussian_kernel, self.gaussian_kernel, 6]
        filtered = gaussian_filter(image, kernel_shape)
        self.logging.info("Filtering completed")

        # Thresholding
        if self.threshold_type.lower() == "otsu":
            thresh = threshold_otsu(filtered)
            self.logging.info(
                "Thresholding with {} threshold type".format(
                    self.threshold_type
                )
            )

        elif (
            self.threshold_type.lower() == "percentile"
            or self.threshold_type.lower() == "perc"
        ):
            thresh = np.percentile(filtered.ravel(), self.percentile_threshold)
            self.logging.info(
                "Thresholding with {} threshold type. {}th percentile [{}]".format(
                    self.threshold_type, self.percentile_threshold, thresh
                )
            )
        else:
            raise ValueError(
                "Unrecognised thresholding type: " + self.threshold_type
            )

        binary = filtered > thresh
        oriented_binary = reorient_image(
            binary, invert_axes=[2,], orientation="coronal"
        )

        # Save thresholded image
        if not os.path.isfile(self.thresholded_savepath) or self.overwrite:
            self.logging.info(
                "Saving thresholded image to {}".format(
                    self.thresholded_savepath
                )
            )
            brainio.to_nii(binary.astype(np.int16), self.thresholded_savepath)

        # apply marching cubes
        self.logging.info("Extracting surface from thresholded image")
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            oriented_binary, 0, step_size=1
        )

        # Scale to atlas spacing
        if voxel_size is not 1:
            verts = verts * voxel_size

        # Save image to .obj
        self.logging.info(" Saving .obj at {}".format(self.obj_path))
        faces = faces + 1
        marching_cubes_to_obj((verts, faces, normals, values), self.obj_path)

        # Keep only the largest connected component
        # DO THIS AT THE IMAGE STAGE
        # get_largest_component(self.obj_path)


def main():
    args = extraction_parser().parse_args()

    # Get output directory
    if args.output_directory is None:
        outdir = os.getcwd()
    elif not os.path.isdir(args.output_directory):
        raise ValueError("Output directory invalid")
    else:
        outdir = args.output_directory

    if args.obj_path is None:
        args.obj_path = Path(args.img_filepath).with_suffix(".obj")
    else:
        args.obj_path = Path(args.obj_path)

    # Start log
    fancylog.start_logging(
        outdir,
        package_for_log,
        filename="injection_finder",
        verbose=args.debug,
        log_to_file=args.save_log,
    )

    # Start extraction
    Extractor(
        args.img_filepath,
        args.registration_folder,
        logging,
        overwrite=args.overwrite,
        gaussian_kernel=args.gaussian_kernel,
        percentile_threshold=args.percentile_threshold,
        threshold_type=args.threshold_type,
        obj_path=args.obj_path,
        overwrite_registration=args.overwrite_registration,
    )


if __name__ == "__main__":
    main()
