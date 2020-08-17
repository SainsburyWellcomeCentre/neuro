import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from skimage.measure import regionprops_table
from vedo import mesh, Spheres, Spline

from imlib.pandas.misc import initialise_df
from imlib.source.source_files import source_custom_config_amap
from imlib.general.list import unique_elements_lists
from imlib.general.system import ensure_directory_exists

from neuro.generic_neuro_tools import save_brain
from neuro.visualise.brainrender_tools import (
    volume_to_vector_array_to_obj_file,
    load_regions_into_brainrender,
)
from neuro.atlas_tools.array import lateralise_atlas
from neuro.atlas_tools.misc import get_voxel_volume, get_atlas_pixel_sizes
from neuro.structures.structures_tree import (
    atlas_value_to_name,
    UnknownAtlasValue,
)
from neuro.visualise.napari_tools.layers import (
    prepare_load_nii,
    add_new_label_layer,
)


def summarise_brain_regions(label_layers, filename):
    summaries = []
    for label_layer in label_layers:
        summaries.append(summarise_single_brain_region(label_layer))

    result = pd.concat(summaries)

    volume_header = "volume_mm3"
    length_columns = [
        "x_min_um",
        "y_min_um",
        "z_min_um",
        "x_max_um",
        "y_max_um",
        "z_max_um",
        "x_center_um",
        "y_center_um",
        "z_center_um",
    ]

    result.columns = ["region"] + [volume_header] + length_columns

    atlas_pixel_sizes = get_atlas_pixel_sizes(source_custom_config_amap())
    voxel_volume = get_voxel_volume(source_custom_config_amap()) / (1000 ** 3)

    result[volume_header] = result[volume_header] * voxel_volume

    for header in length_columns:
        for dim in atlas_pixel_sizes.keys():
            if header.startswith(dim):
                scale = float(atlas_pixel_sizes[dim])
        assert scale > 0

        result[header] = result[header] * scale

    result.to_csv(filename, index=False)


def summarise_single_brain_region(
    label_layer,
    ignore_empty=True,
    properties_to_fetch=["area", "bbox", "centroid",],
):
    data = label_layer.data
    if ignore_empty:
        if data.sum() == 0:
            return

    # swap data back to original orientation from napari orientation
    data = np.swapaxes(data, 2, 0)

    regions_table = regionprops_table(data, properties=properties_to_fetch)
    df = pd.DataFrame.from_dict(regions_table)
    df.insert(0, "Region", label_layer.name)
    return df


def add_existing_track_layers(
    viewer, track_file, point_size, x_scaling, y_scaling, z_scaling
):
    max_z = len(viewer.layers[0].data)
    data = brainrender_track_to_napari(
        track_file, x_scaling, y_scaling, z_scaling, max_z
    )
    new_points_layer = viewer.add_points(
        data, n_dimensional=True, size=point_size, name=Path(track_file).stem,
    )
    new_points_layer.mode = "ADD"
    return new_points_layer


def brainrender_track_to_napari(
    track_file, x_scaling, y_scaling, z_scaling, max_z
):
    points = pd.read_hdf(track_file)
    points["x"] = points["x"] / z_scaling
    points["z"] = points["z"] / x_scaling
    points["y"] = points["y"] / y_scaling

    points["x"] = max_z - points["x"]

    return points.to_numpy().astype(np.int16)


def add_existing_label_layers(
    viewer,
    label_file,
    selected_label=1,
    num_colors=10,
    brush_size=30,
    memory=False,
):
    """
    Loads an existing (nii) image as a napari labels layer
    :param viewer: Napari viewer instance
    :param label_file: Filename of the image to be loaded
    :param int selected_label: Label ID to be preselected
    :param int num_colors: How many colors (labels)
    :param int brush_size: Default size of the label brush
    :return label_layer: napari labels layer
    """
    label_file = Path(label_file)
    labels = prepare_load_nii(label_file, memory=memory)
    label_layer = viewer.add_labels(
        labels, num_colors=num_colors, name=label_file.stem
    )
    label_layer.selected_label = selected_label
    label_layer.brush_size = brush_size
    return label_layer


def save_regions_to_file(
    label_layer,
    destination_directory,
    template_image,
    ignore_empty=True,
    obj_ext=".obj",
    image_extension=".nii",
):
    """
    Analysed the regions (to see what brain areas they are in) and saves
    the segmented regions to file (both as .obj and .nii)
    :param label_layer: napari labels layer (with segmented regions)
    :param destination_directory: Where to save files to
    :param template_image: Existing image of size/shape of the
    destination images
    the values in "annotations" and a "name column"
    :param ignore_empty: If True, don't attempt to save empty images
    :param obj_ext: File extension for the obj files
    :param image_extension: File extension fo the image files
    """
    data = label_layer.data
    if ignore_empty:
        if data.sum() == 0:
            return

    # swap data back to original orientation from napari orientation
    data = np.swapaxes(data, 2, 0)
    name = label_layer.name

    filename = destination_directory / (name + obj_ext)
    volume_to_vector_array_to_obj_file(
        data, filename,
    )

    filename = destination_directory / (name + image_extension)
    save_brain(
        data, template_image, filename,
    )


def analyse_region_brain_areas(
    label_layer,
    destination_directory,
    annotations,
    hemispheres,
    structures_reference_df,
    extension=".csv",
    ignore_empty=True,
):
    """

    :param label_layer: napari labels layer (with segmented regions)
    :param np.array annotations: numpy array of the brain area annotations
    :param np.array hemispheres: numpy array of hemipshere annotations
    :param structures_reference_df: Pandas dataframe with "id" column (matching
    the values in "annotations" and a "name column"
    :param ignore_empty: If True, don't analyse empty regions
    """

    data = label_layer.data
    if ignore_empty:
        if data.sum() == 0:
            return

    # swap data back to original orientation from napari orientation
    data = np.swapaxes(data, 2, 0)
    name = label_layer.name

    masked_annotations = data.astype(bool) * annotations

    # TODO: don't hardcode hemisphere value. Get from atlas config
    annotations_left, annotations_right = lateralise_atlas(
        masked_annotations,
        hemispheres,
        left_hemisphere_value=2,
        right_hemisphere_value=1,
    )

    unique_vals_left, counts_left = np.unique(
        annotations_left, return_counts=True
    )
    unique_vals_right, counts_right = np.unique(
        annotations_right, return_counts=True
    )

    voxel_volume = get_voxel_volume(source_custom_config_amap())
    voxel_volume_in_mm = voxel_volume / (1000 ** 3)

    df = initialise_df(
        "structure_name",
        "left_volume_mm3",
        "left_percentage_of_total",
        "right_volume_mm3",
        "right_percentage_of_total",
        "total_volume_mm3",
        "percentage_of_total",
    )

    sampled_structures = unique_elements_lists(
        list(unique_vals_left) + list(unique_vals_right)
    )
    total_volume_region = get_total_volume_regions(
        unique_vals_left, unique_vals_right, counts_left, counts_right
    )

    for atlas_value in sampled_structures:
        if atlas_value != 0:
            try:
                df = add_structure_volume_to_df(
                    df,
                    atlas_value,
                    structures_reference_df,
                    unique_vals_left,
                    unique_vals_right,
                    counts_left,
                    counts_right,
                    voxel_volume_in_mm,
                    total_volume_voxels=total_volume_region,
                )

            except UnknownAtlasValue:
                print(
                    "Value: {} is not in the atlas structure reference file. "
                    "Not calculating the volume".format(atlas_value)
                )
    filename = destination_directory / (name + extension)
    df.to_csv(filename, index=False)


def get_total_volume_regions(
    unique_vals_left, unique_vals_right, counts_left, counts_right,
):
    zero_index_left = np.where(unique_vals_left == 0)[0][0]
    counts_left = list(counts_left)
    counts_left.pop(zero_index_left)

    zero_index_right = np.where(unique_vals_right == 0)[0][0]
    counts_right = list(counts_right)
    counts_right.pop(zero_index_right)

    return sum(counts_left + counts_right)


def add_structure_volume_to_df(
    df,
    atlas_value,
    structures_reference_df,
    unique_vals_left,
    unique_vals_right,
    counts_left,
    counts_right,
    voxel_volume,
    total_volume_voxels=None,
):
    name = atlas_value_to_name(atlas_value, structures_reference_df)

    left_volume, left_percentage = get_volume_in_hemisphere(
        atlas_value,
        unique_vals_left,
        counts_left,
        total_volume_voxels,
        voxel_volume,
    )
    right_volume, right_percentage = get_volume_in_hemisphere(
        atlas_value,
        unique_vals_right,
        counts_right,
        total_volume_voxels,
        voxel_volume,
    )
    if total_volume_voxels is not None:
        total_percentage = left_percentage + right_percentage
    else:
        total_percentage = 0

    df = df.append(
        {
            "structure_name": name,
            "left_volume_mm3": left_volume,
            "left_percentage_of_total": left_percentage,
            "right_volume_mm3": right_volume,
            "right_percentage_of_total": right_percentage,
            "total_volume_mm3": left_volume + right_volume,
            "percentage_of_total": total_percentage,
        },
        ignore_index=True,
    )
    return df


def get_volume_in_hemisphere(
    atlas_value, unique_vals, counts, total_volume_voxels, voxel_volume
):
    try:
        index = np.where(unique_vals == atlas_value)[0][0]
        volume = counts[index] * voxel_volume
        if total_volume_voxels is not None:
            percentage = 100 * (counts[index] / total_volume_voxels)
        else:
            percentage = 0
    except IndexError:
        volume = 0
        percentage = 0

    return volume, percentage


def convert_and_save_points(
    points_layers,
    output_directory,
    x_scaling,
    y_scaling,
    z_scaling,
    max_z,
    track_file_extension=".h5",
):
    """
    Converts the points from the napari format (in image space) to brainrender
    (in atlas space)
    :param points_layers: list of points layers
    :param output_directory: path to save points to
    :param x_scaling: scaling from image space to brainrender scene
    :param y_scaling: scaling from image space to brainrender scene
    :param z_scaling: scaling from image space to brainrender scene
    :param max_z: Maximum extent of the image in z
    """

    output_directory = Path(output_directory)
    ensure_directory_exists(output_directory)

    for points_layer in points_layers:
        save_single_track_layer(
            points_layer,
            output_directory,
            x_scaling,
            y_scaling,
            z_scaling,
            max_z,
            track_file_extension=track_file_extension,
        )


def save_single_track_layer(
    layer,
    output_directory,
    x_scaling,
    y_scaling,
    z_scaling,
    max_z,
    track_file_extension=".h5",
):
    output_filename = output_directory / (layer.name + track_file_extension)
    cells = layer.data.astype(np.int16)
    cells = pd.DataFrame(cells)

    cells.columns = ["x", "y", "z"]

    # weird scaling due to the ARA coordinate space
    cells["x"] = max_z - cells["x"]
    cells["x"] = z_scaling * cells["x"]
    cells["z"] = x_scaling * cells["z"]
    cells["y"] = y_scaling * cells["y"]
    cells.to_hdf(output_filename, key="df", mode="w")


def analyse_track(
    scene,
    points_file,
    add_surface_to_points=True,
    spline_points=100,
    fit_degree=3,
    spline_smoothing=0.05,
    point_radius=30,
    spline_radius=10,
    verbose=True,
):
    """
    Given a file of points, fit a spline function, and add to a brainrender
     scene.
    :param scene: brainrender scene object
    :param points_file:
    :param bool add_surface_to_points: Add the closest part of the brain
    surface to the list of points
    :param spline_points: How many points define the spline
    :param fit_degree: spline fit degree
    :param spline_smoothing: spline fit smoothing
    :param point_radius: size of the points in the brainrender scene
    :param spline_radius: size of the rendered spline in the brainrender
    scene
    :param bool verbose: Whether to print the progress
    :return:
        scene: brainrender scene with the surface point added.
        spline: vedo spline object
    """
    points = pd.read_hdf(points_file)
    scene.add_cells(
        points,
        color_by_region=True,
        res=12,
        radius=point_radius,
        verbose=False,
    )
    points = np.array(points)

    if add_surface_to_points:
        scene, points = add_surface_point_to_points(
            scene, points, point_radius, verbose=verbose
        )

    far_point = np.expand_dims(points[-1], axis=0)
    scene.add_vtkactor(Spheres(far_point, r=point_radius).color("n"))

    spline = (
        Spline(
            points,
            smooth=spline_smoothing,
            degree=fit_degree,
            res=spline_points,
        )
        .pointSize(spline_radius)
        .color("n")
    )

    return scene, spline


def add_surface_point_to_points(
    scene, points, point_radius, color="n", verbose=True
):
    """
    Adds the closest part of the brain surface to the list of points. Returns
    the brainrender scene with the point added, and the point added to the
    list of points
    :param scene: brainrender scene object
    :param points: List of points
    :param point_radius: Radius of the point when displayed
    :param bool verbose: Whether to print the progress
    :return:
        scene: brainrender scene with the surface point added.
        points: list of points with the surface point added.
    """
    if verbose:
        print(
            "Finding the closest point on the brain surface to the first point"
        )
    root_mesh = mesh.Mesh(scene.root)
    surface_intersection = np.expand_dims(
        root_mesh.closestPoint(points[0]), axis=0
    )
    points = np.concatenate([surface_intersection, points], axis=0)
    scene.add_vtkactor(
        Spheres(surface_intersection, r=point_radius).color(color)
    )
    return scene, points


def analyse_track_anatomy(scene, spline, file_path, verbose=True):
    """
    For a given spline, and brainrender scene, find the brain region that each
    "segment" is in, and save to csv.

    :param scene: brainrender scene object
    :param spline: vtkplotter spline object
    :param file_path: path to save the results to
    :param bool verbose: Whether to print the progress
    """
    if verbose:
        print("Determining the brain region for each segment of the spline")
    spline_regions = [
        scene.atlas.get_structure_from_coordinates(p, just_acronym=False)
        for p in spline.points().tolist()
    ]

    df = pd.DataFrame(
        columns=["Position", "Region ID", "Region acronym", "Region name"]
    )
    for idx, spline_region in enumerate(spline_regions):
        if spline_region is None:
            df = df.append(
                {
                    "Position": idx,
                    "Region ID": "Not found in brain",
                    "Region acronym": "Not found in brain",
                    "Region name": "Not found in brain",
                },
                ignore_index=True,
            )
        else:
            df = df.append(
                {
                    "Position": idx,
                    "Region ID": spline_region["id"],
                    "Region acronym": spline_region["acronym"],
                    "Region name": spline_region["name"],
                },
                ignore_index=True,
            )
    if verbose:
        print(f"Saving results to: {file_path}")
    df.to_csv(file_path, index=False)


def display_track_in_brainrender(
    scene, spline, regions_to_add=[], region_alpha=0.3, verbose=True
):
    """

    :param scene: brainrender scene object
    :param spline: vtkplotter spline object
    :param regions_to_add: List of additional brain regions to add, as a list
    of acronyms
    :param region_alpha: Opacity of the displayed regions
    :param bool verbose: Whether to print the progress
    """
    if verbose:
        print("Visualising 3D data in brainrender")
    scene.add_vtkactor(spline)
    scene.add_brain_regions(regions_to_add, alpha=region_alpha)
    scene.verbose = False
    return scene


def view_in_brainrender(
    scene,
    spline,
    regions_directory,
    alpha=0.8,
    shading="flat",
    region_to_add=[],
    region_alpha=0.3,
):
    obj_files = glob(str(regions_directory) + "/*.obj")
    if obj_files:
        scene = load_regions_into_brainrender(
            scene, obj_files, alpha=alpha, shading=shading
        )
    try:
        scene = display_track_in_brainrender(
            scene,
            spline,
            regions_to_add=region_to_add,
            region_alpha=region_alpha,
        )
    except:
        pass

    scene.render()


def add_new_track_layer(viewer, track_layers, point_size):
    num = len(track_layers)
    new_track_layers = viewer.add_points(
        n_dimensional=True, size=point_size, name=f"track_{num}",
    )
    new_track_layers.mode = "ADD"
    track_layers.append(new_track_layers)


def add_new_region_layer(
    viewer, label_layers, image_like, brush_size, num_colors
):
    num = len(label_layers)
    new_label_layer = add_new_label_layer(
        viewer,
        image_like,
        name=f"region_{num}",
        brush_size=brush_size,
        num_colors=num_colors,
    )
    new_label_layer.mode = "PAINT"
    label_layers.append(new_label_layer)


def add_existing_region_segmentation(
    directory, viewer, label_layers, file_extension, memory=False
):
    label_files = glob(str(directory) + "/*" + file_extension)
    if directory and label_files != []:
        for label_file in label_files:
            label_layers.append(
                add_existing_label_layers(viewer, label_file, memory=memory)
            )
