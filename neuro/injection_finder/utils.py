import numpy as np
from vtkplotter.analysis import extractLargestRegion

# ----------------------- IMAGE MANIPULATION FUNCTIONS ----------------------- #
def reorient_image(image, invert_axes=None, orientation="saggital"):
    """
    Reorients the image to the coordinate space of the atlas

    :param image_path: str
    :param threshold: float
    :param invert_axes: tuple (Default value = None)
    :param image: 
    :param orientation:  (Default value = "saggital")

    """
    if invert_axes is not None:
        for axis in list(invert_axes):
            image = np.flip(image, axis=axis)

    if orientation is not "saggital":
        if orientation is "coronal":
            transposition = (2, 1, 0)
        elif orientation is "horizontal":
            transposition = (1, 2, 0)

        image = np.transpose(image, transposition)
    return image


# ------------------------- MARCHING CUBES FUNCTIONS ------------------------- #
def marching_cubes_to_obj(marching_cubes_out, output_file):
    """
    Saves the output of skimage.measure.marching_cubes as an .obj file

    :param marching_cubes_out: tuple
    :param output_file: str

    """

    verts, faces, normals, _ = marching_cubes_out
    with open(output_file, "w") as f:
        for item in verts:
            f.write(f"v {item[0]} {item[1]} {item[2]}\n")
        for item in normals:
            f.write(f"vn {item[0]} {item[1]} {item[2]}\n")
        for item in faces:
            f.write(
                f"f {item[0]}//{item[0]} {item[1]}//{item[1]} "
                f"{item[2]}//{item[2]}\n"
            )
        f.close()


# ----------------------------- VTKPLOTTER UTILS ----------------------------- #
def get_center_of_mass(actor):
    """
        Get the center of mass of a vtk actor
    """
    return actor.centerOfMass()


def get_largest_component(obj_filepath):
    """
        Given a .obj file with multiple disconnected meshes in it, it
        selects the largest of these and discards the rest. 
    """
    actor = load(obj_filepath)
    actor = extractLargestRegion(actor).flipNormals()
    save(actor, obj_filepath)
