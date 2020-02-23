from brainio import brainio
from brainrender.Utils.image import reorient_image, marching_cubes_to_obj
from brainrender.scene import Scene
from skimage import measure
import numpy as np

from lesion_estimation.brain_tools.atlas_structures import (
    PATH_TO_ATLAS_IMAGE,
    smooth_structure,
)


def volume_to_vector_array_to_obj_file(image, output_path, voxel_size=10):

    oriented_binary = reorient_image(
        image, invert_axes=[2], orientation="coronal"
    )

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        oriented_binary, 0, step_size=1
    )

    if voxel_size is not 1:
        verts = verts * voxel_size

    faces = faces + 1
    marching_cubes_to_obj((verts, faces, normals, values), str(output_path))


def visualize_obj(obj_path, *args, color="lightcoral", **kwargs):
    """
        Uses brainrender to visualize a .obj file registered to the Allen CCF
        :param obj_path: str, path to a .obj file
        :param color: str, color of object being rendered
    """
    print("Visualizing : " + obj_path)
    scene = Scene(add_root=True)
    scene.add_from_file(obj_path, *args, c=color, **kwargs)

    return scene


def create_scene(default_structures):
    scene = Scene(add_root=True)
    for structure in default_structures:
        scene.add_brain_regions([structure], use_original_color=True)
    return scene


def make_smoothed_atlas_region(
    output_dir,
    atlas_ids,
    structure_name,
    atlas_path=PATH_TO_ATLAS_IMAGE,
    smoothing_threshold=0.4,
    sigma=10,
    voxel_size=10,
):

    atlas = brainio.load_any(atlas_path)

    all_regions = np.zeros_like(atlas)
    for id in atlas_ids:
        region_mask = atlas == id
        all_regions = np.logical_or(region_mask, all_regions)

    all_regions = smooth_structure(
        all_regions, threshold=smoothing_threshold, sigma=sigma
    )

    output_path = f"{output_dir}{structure_name}.obj"
    oriented_binary = reorient_image(
        all_regions, invert_axes=[2], orientation="coronal"
    )

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        oriented_binary, 0, step_size=1
    )

    if voxel_size is not 1:
        verts = verts * voxel_size

    faces = faces + 1
    marching_cubes_to_obj((verts, faces, normals, values), output_path)
