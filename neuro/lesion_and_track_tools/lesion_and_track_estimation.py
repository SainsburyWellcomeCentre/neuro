import os
import pathlib

import numpy as np
import skimage
from skimage.segmentation import flood
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import (
    gaussian,
    threshold_otsu,
    median,
    threshold_multiotsu,
)

from brainio import brainio
from neuro.generic_neuro_tools import (
    save_brain,
    transform_all_channels_to_standard_space,
)


def get_fiber_track(
    registered_atlas_path,
    source_image_path,
    seed_point,
    output_path=None,
    normalising_factor=4,
    erosion_selem=(5, 5, 5),
):

    """
    gets segmentation image of optical fibers using the background channel and a seed-point,
    tested on 200um and 400um fibers

    :param erosion_selem:
    :param normalising_factor:
    :param output_path:
    :param registered_atlas_path:
    :param source_image_path:
    :param seed_point:
    :return:
    """
    brain_mask = brainio.load_any(str(registered_atlas_path)) != 0
    brain = brainio.load_any(str(source_image_path))

    brain_median_filtered = median(brain)
    otsu_threshold = threshold_multiotsu(brain_median_filtered)[0]
    brain_segmentation = (
        brain_median_filtered < otsu_threshold / normalising_factor
    ) * brain_mask

    segmentation_eroded = binary_erosion(
        brain_segmentation, selem=erosion_selem
    )
    segmentation_eroded_dilated = binary_dilation(segmentation_eroded)
    fiber_track_image = flood(
        segmentation_eroded_dilated.astype(np.int16), seed_point
    )

    if output_path is not None:
        save_brain(fiber_track_image, source_image_path, output_path)
    else:
        return fiber_track_image


def get_lesion(
    reg_dir,
    allen_structure_id=672,
    erosion_selem=np.ones((5, 5, 5)),
    lesion_threshold=0.5,
    sigma=5,
    minimum_object_size=8000,
    otsu_scale_factor=2,
    transformed_brain_file_fname="registered_downsampled_channel_0.nii",
    atlas_fname="annotations.nii",
    output_fname="lesion_mask.nii",
):
    """

    :param atlas_fname:
    :param otsu_scale_factor:
    :param transformed_brain_file_fname:
    :param minimum_object_size: any blobs below this size will be ignored
    :param sigma: sigma for gaussian filtering
    :param lesion_threshold:
    :param reg_dir: directory containing cellfinder/amap registration output files
    :param allen_structure_id: the id of the structure that contains the lesion
    :param erosion_selem:
    :return:
    """
    reg_dir = pathlib.Path(reg_dir)
    annotations_path = reg_dir / atlas_fname
    source_image_path = reg_dir / transformed_brain_file_fname

    if not os.path.isfile(str(source_image_path)):
        print("channel not in std space... gotta transform")
        transform_all_channels_to_standard_space(reg_dir)

    brain_mask = brainio.load_any(str(annotations_path)) == allen_structure_id
    brain = brainio.load_any(str(source_image_path))

    brain_otsu_thresh = threshold_otsu(brain)
    brain = brain < (brain_otsu_thresh / otsu_scale_factor)
    brain = brain * brain_mask
    brain = gaussian(brain, sigma) > lesion_threshold
    brain = binary_erosion(brain, selem=erosion_selem)
    brain = skimage.morphology.label(brain)
    biggest = skimage.morphology.remove_small_objects(
        brain, min_size=minimum_object_size
    )
    save_brain(biggest, annotations_path, reg_dir / output_fname)


def get_lesion_sizes(reg_dir):
    reg_dir = pathlib.Path(reg_dir)
    labels = brainio.load_any(str(reg_dir / "lesion_mask.nii"))
    all_structures_ids = np.unique(labels)
    return [np.count_nonzero(labels == idx) for idx in all_structures_ids]
