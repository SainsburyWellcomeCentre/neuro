import numpy as np
from pathlib import Path
from brainrender.Utils.image import reorient_image, marching_cubes_to_obj
from brainrender.scene import Scene
from skimage import measure

from imlib.general.pathlib import append_to_pathlib_stem
from imlib.plotting.colors import get_random_vtkplotter_color

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


def volume_to_vector_array_to_obj_file(
    image,
    output_path,
    invert_axes=[2],
    voxel_size=10,
    orientation="coronal",
    step_size=1,
    threshold=0,
    deal_with_regions_separately=False,
):

    oriented_binary = reorient_image(
        image, invert_axes=invert_axes, orientation=orientation
    )

    if deal_with_regions_separately:
        for label_id in np.unique(oriented_binary):
            if label_id != 0:
                filename = append_to_pathlib_stem(
                    Path(output_path), "_" + str(label_id)
                )
                image = oriented_binary == label_id
                extract_and_save_object(
                    image,
                    filename,
                    voxel_size=voxel_size,
                    threshold=threshold,
                    step_size=step_size,
                )
    else:
        extract_and_save_object(
            oriented_binary,
            output_path,
            voxel_size=voxel_size,
            threshold=threshold,
            step_size=step_size,
        )


def extract_and_save_object(
    image, output_file_name, voxel_size=10, threshold=0, step_size=1
):
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        image, threshold, step_size=step_size
    )
    verts, faces = convert_obj_to_br(verts, faces, voxel_size=voxel_size)
    marching_cubes_to_obj(
        (verts, faces, normals, values), str(output_file_name)
    )


def convert_obj_to_br(verts, faces, voxel_size=10):
    if voxel_size is not 1:
        verts = verts * voxel_size

    faces = faces + 1
    return verts, faces


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


def load_regions_into_brainrender(list_of_regions, alpha=0.8, shading="flat"):
    """
    Loads a list of .obj files into brainrender
    :param list_of_regions: List of .obj files to be loaded
    :param alpha: Object transparency
    :param shading: Object shading type ("flat", "giroud" or "phong").
    Defaults to "phong"
    """
    scene = Scene()
    for obj_file in list_of_regions:
        load_obj_into_brainrender(
            scene, obj_file, alpha=alpha, shading=shading
        )
    scene.render()


def load_obj_into_brainrender(
    scene, obj_file, color=None, alpha=0.8, shading="phong"
):
    """
    Loads a single obj file into brainrender
    :param scene: brainrender scene
    :param obj_file: obj filepath
    :param color: Object color. If None, a random color is chosen
    :param alpha: Object transparency
    :param shading: Object shading type ("flat", "giroud" or "phong").
    Defaults to "phong"
    """
    obj_file = str(obj_file)
    if color is None:
        color = get_random_vtkplotter_color()
    act = scene.add_from_file(obj_file, c=color, alpha=alpha)

    if shading == "flat":
        act.GetProperty().SetInterpolationToFlat()
    elif shading == "gouraud":
        act.GetProperty().SetInterpolationToGouraud()
    else:
        act.GetProperty().SetInterpolationToPhong()


def create_scene(default_structures):
    scene = Scene(add_root=True)
    for structure in default_structures:
        scene.add_brain_regions([structure], use_original_color=True)
    return scene
