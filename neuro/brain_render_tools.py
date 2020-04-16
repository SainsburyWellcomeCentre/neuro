from brainrender.Utils.image import reorient_image, marching_cubes_to_obj
from brainrender.scene import Scene
from skimage import measure

from neuro.atlas_tools.custom_atlas_structures import (
    get_arbitrary_structure_mask_from_custom_atlas,
)


def render_region_from_custom_atlas(
    output_dir,
    atlas_ids,
    structure_name,
    atlas_path,
    smoothing_threshold=0.4,
    sigma=10,
    voxel_size=10,
):

    all_regions = get_arbitrary_structure_mask_from_custom_atlas(
        atlas_ids, atlas_path, sigma, smoothing_threshold
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
