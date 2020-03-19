"""
Based on cellfinder detected cells, and amap registration, generate a heatmap
image
"""

import logging
import argparse

import numpy as np
from skimage.filters import gaussian
from skimage.transform import resize
from scipy.ndimage import zoom
from brainio import brainio
from imlib.cells.utils import get_cell_location_array
from imlib.image.scale import scale_and_convert_to_16_bits
from imlib.image.binning import get_bins
from imlib.image.shape import convert_shape_dict_to_array_shape
from imlib.image.masking import mask_image_threshold
from imlib.general.numerical import check_positive_float


def run(
    cells_file,
    output_filename,
    target_size,
    raw_image_shape,
    raw_image_bin_sizes,
    transformation_matrix,
    atlas_scale,
    smoothing=10,
    mask=True,
    atlas=None,
    cells_only=True,
    convert_16bit=True,
):
    """

    :param cells_file: Cellfinder output cells file.
    :param output_filename: File to save heatmap into
    :param target_size: Size of the final heatmap
    :param raw_image_shape: Size of the raw data (coordinate space of the
    cells)
    :param raw_image_bin_sizes: List/tuple of the sizes of the bins in the
    raw data space
    :param transformation_matrix: Transformation matrix so that the resulting
    nifti can be processed using other tools.
    :param atlas_scale: Image scaling so that the resulting nifti can be
    processed using other tools.
    :param smoothing: Smoothing kernel size, in the target image space
    :param mask: Whether or not to mask the heatmap based on an atlas file
    :param atlas: Atlas file to mask the heatmap
    :param cells_only: Only use "cells", not artefacts
    :param convert_16bit: Convert final image to 16 bit


    """

    # TODO: compare the smoothing effects of gaussian filtering, and upsampling
    target_size = convert_shape_dict_to_array_shape(target_size, type="fiji")
    raw_image_shape = convert_shape_dict_to_array_shape(
        raw_image_shape, type="fiji"
    )
    cells_array = get_cell_location_array(cells_file, cells_only=cells_only)
    bins = get_bins(raw_image_shape, raw_image_bin_sizes)

    logging.debug("Generating heatmap (3D histogram)")
    heatmap_array, _ = np.histogramdd(cells_array, bins=bins)
    # otherwise resized array is too big to fit into RAM
    heatmap_array = heatmap_array.astype(np.uint16)

    logging.debug("Resizing heatmap to the size of the target image")
    # heatmap_array = imresize(heatmap_array, target_size, interp='nearest')

    factors = np.asarray(target_size, dtype=float) / np.asarray(
        heatmap_array.shape, dtype=float
    )

    heatmap_array = zoom(heatmap_array, factors, order=0)

    if smoothing is not None:
        logging.debug(
            "Applying Gaussian smoothing with a kernel sigma of: "
            "{}".format(smoothing)
        )
        heatmap_array = gaussian(heatmap_array, sigma=smoothing)

    if mask:
        logging.debug("Masking image based on registered atlas")
        # copy, otherwise it's modified, which affects later figure generation
        atlas_for_mask = np.copy(atlas)
        heatmap_array = mask_image_threshold(heatmap_array, atlas_for_mask)

    if convert_16bit:
        logging.debug("Converting to 16 bit")
        heatmap_array = scale_and_convert_to_16_bits(heatmap_array)

    logging.debug("Saving heatmap image")
    brainio.to_nii(
        heatmap_array,
        output_filename,
        scale=atlas_scale,
        affine_transform=transformation_matrix,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="cells_file", type=str, help="Cellfinder output cell file",
    )
    parser.add_argument(
        dest="output_filename",
        type=str,
        help="Output filename. Should end with '.nii'",
    )

    parser.add_argument(
        dest="raw_image", type=str, help="Paths to raw data",
    )

    parser.add_argument(
        dest="downsampled_image", type=str, help="Downsampled_atlas .nii file",
    )
    parser.add_argument(
        "--bin-size",
        dest="bin_size_um",
        type=check_positive_float,
        default=100,
        help="Heatmap bin size (um of each edge of histogram cube)",
    )
    parser.add_argument(
        "-x",
        "--x-pixel-um",
        dest="x_pixel_um",
        type=check_positive_float,
        help="Pixel spacing of the data in the first "
        "dimension, specified in um.",
    )
    parser.add_argument(
        "-y",
        "--y-pixel-um",
        dest="y_pixel_um",
        type=check_positive_float,
        help="Pixel spacing of the data in the second "
        "dimension, specified in um.",
    )
    parser.add_argument(
        "-z",
        "--z-pixel-um",
        dest="z_pixel_um",
        type=check_positive_float,
        help="Pixel spacing of the data in the third "
        "dimension, specified in um.",
    )
    parser.add_argument(
        "--heatmap-smoothing",
        dest="heatmap_smooth",
        type=check_positive_float,
        default=100,
        help="Gaussian smoothing sigma, in um.",
    )
    parser.add_argument(
        "--no-mask-figs",
        dest="mask_figures",
        action="store_false",
        help="Don't mask the figures (removing any areas outside the brain,"
        "from e.g. smoothing)",
    )

    return parser


class HeatmapParams:
    # assumes an isotropic target space
    def __init__(
        self,
        raw_image,
        downsampled_image,
        bin_size_um,
        x_pixel_um,
        y_pixel_um,
        z_pixel_um,
        smoothing_target_space,
    ):
        self._input_image = raw_image
        self._target_image = downsampled_image
        self._bin_um = bin_size_um
        self._x_pixel_um = x_pixel_um
        self._y_pixel_um = y_pixel_um
        self._z_pixel_um = z_pixel_um
        self._smooth_um = smoothing_target_space
        self._downsampled_image = None

        self.figure_image_shape = None
        self.raw_image_shape = None
        self.bin_size_raw_voxels = None
        self.atlas_scale = None
        self.transformation_matrix = None
        self.smoothing_target_voxel = None

        self._get_raw_image_shape()
        self._get_figure_image_shape()
        self._get_atlas_data()
        self._get_atlas_scale()
        self._get_transformation_matrix()
        self._get_binning()
        self._get_smoothing()

    def _get_raw_image_shape(self):
        logging.debug("Checking raw image size")
        self.raw_image_shape = brainio.get_size_image_from_file_paths(
            self._input_image
        )
        logging.debug(f"Raw image size: {self.raw_image_shape}")

    def _get_figure_image_shape(self):
        logging.debug(
            "Loading file: {} to check target image size"
            "".format(self._target_image)
        )
        self._downsampled_image = brainio.load_nii(
            self._target_image, as_array=False
        )
        shape = self._downsampled_image.shape
        self.figure_image_shape = {"x": shape[0], "y": shape[1], "z": shape[2]}
        logging.debug("Target image size: {}".format(self.figure_image_shape))

    def _get_binning(self):
        logging.debug("Calculating bin size in raw image space voxels")
        bin_raw_x = int(self._bin_um / self._x_pixel_um)
        bin_raw_y = int(self._bin_um / self._y_pixel_um)
        bin_raw_z = int(self._bin_um / self._z_pixel_um)
        self.bin_size_raw_voxels = [bin_raw_x, bin_raw_y, bin_raw_z]
        logging.debug(
            f"Bin size in raw image space is x:{bin_raw_x}, "
            f"y:{bin_raw_y}, z:{bin_raw_z}."
        )

    def _get_atlas_data(self):
        self.atlas_data = brainio.load_nii(self._target_image, as_array=True)

    def _get_atlas_scale(self):
        self.atlas_scale = self._downsampled_image.header.get_zooms()

    def _get_transformation_matrix(self):
        self.transformation_matrix = self._downsampled_image.affine

    def _get_smoothing(self):
        logging.debug(
            "Calculating smoothing in target image volume. Assumes "
            "an isotropic target image"
        )
        if self._smooth_um is not 0:
            # 1000 is to scale to um
            self.smoothing_target_voxel = int(
                self._smooth_um / (self.atlas_scale[0] * 1000)
            )


def main(
    cells_file,
    output_filename,
    raw_image,
    downsampled_image,
    bin_size_um,
    x_pixel_um,
    y_pixel_um,
    z_pixel_um,
    heatmap_smooth,
    masking,
):
    params = HeatmapParams(
        raw_image,
        downsampled_image,
        bin_size_um,
        x_pixel_um,
        y_pixel_um,
        z_pixel_um,
        heatmap_smooth,
    )

    run(
        cells_file,
        output_filename,
        params.figure_image_shape,
        params.raw_image_shape,
        params.bin_size_raw_voxels,
        params.transformation_matrix,
        params.atlas_scale,
        smoothing=params.smoothing_target_voxel,
        mask=masking,
        atlas=params.atlas_data,
    )


def cli():
    args = get_parser().parse_args()
    main(
        args.cells_file,
        args.output_filename,
        args.raw_image,
        args.downsampled_image,
        args.bin_size_um,
        args.x_pixel_um,
        args.y_pixel_um,
        args.z_pixel_um,
        args.heatmap_smooth,
        args.mask_figures,
    )


if __name__ == "__main__":
    cli()
