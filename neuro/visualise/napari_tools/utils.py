import numpy as np


def convert_vtk_spline_to_napari_path(
    spline, x_scaling, y_scaling, z_scaling, max_z
):
    """
    Converts a vtkplotter spline object to points that can be rendered in
    napari
    :param spline: vtkplotter spline object
    :param x_scaling: scaling from image space to brainrender scene
    :param y_scaling: scaling from image space to brainrender scene
    :param z_scaling: scaling from image space to brainrender scene
    :param max_z: Maximum extent of the image in z
    :return: np.array of spline points
    """
    napari_spline = np.copy(spline.points())
    napari_spline[:, 0] = (z_scaling * max_z - napari_spline[:, 0]) / z_scaling
    napari_spline[:, 1] = napari_spline[:, 1] / y_scaling
    napari_spline[:, 2] = napari_spline[:, 2] / x_scaling
    return napari_spline.astype(np.int16)
