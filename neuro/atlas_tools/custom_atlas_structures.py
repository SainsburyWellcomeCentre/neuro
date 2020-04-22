import pandas as pd
import numpy as np
from brainio import brainio
from skimage.filters import gaussian

from neuro.visualise import brainrender


def load_atlas_structures_csv(path_to_structures_csv):
    df = pd.read_csv(path_to_structures_csv)
    return df


def get_atlas_ids(df, parent_key):
    ids = [int(id) for id in df[df["parent_id"] == parent_key]["id"]]
    return ids


def create_hierarchy_paths(df):
    """
    creates paths of id hierarchy to match custom atlas to allen atlas
    :param df:
    :return:
    """
    all_paths = {}
    all_ids = df["id"]
    all_ids = all_ids[all_ids.notnull()]
    for id in all_ids:
        path = get_structure_parents(df, id)
        path = get_path_string_standard_fmt(path)
        all_paths.setdefault(id, path)
    return all_paths


def get_structure_parents(df, k, parent=None, all_parents=None, root_id=997):
    """
    gets all parent structures of the given id

    :param df:
    :param k:
    :param parent:
    :param all_parents:
    :param root_id:
    :return:
    """
    if parent is None:
        all_parents = []
        parent = df[df["id"] == k]["parent_id"].values[0]
        all_parents.append(parent)

    if int(parent) != root_id:
        parent = df[df["id"] == parent]["parent_id"].values[0]
        all_parents.append(parent)
        return get_structure_parents(df, k, parent, all_parents)
    return all_parents[::-1]


def get_path_string_standard_fmt(all_parent_ids):
    all_parent_ids = [str(i) for i in all_parent_ids]
    return "/".join(all_parent_ids)


def add_to_df(df, df_dict):
    df.insert(len(df.keys()), "structure_id_path", np.full(len(df), np.nan))
    for k, v in df_dict.items():
        loc = df[df["id"] == k].index[0]
        df["structure_id_path"].iloc[loc] = v
    return df


def get_all_structure_children(df, k):
    all_ids = df["id"]
    all_ids = all_ids[all_ids.notnull()]
    all_children = []
    for id in all_ids:
        all_parents = get_structure_parents(df, id)
        if k in all_parents:
            all_children.append(id)
    if len(all_children) == 0:
        all_children.append(k)
    return all_children


def render_all_subregions(
    atlas_id,
    out_dir,
    atlas_path,
    structures_csv_path,
    smooth_threshold=0.4,
    smooth_sigma=10,
):
    """
    renders all children structures of a given id

    :param atlas_id:
    :param out_dir:
    :param atlas_path:
    :param structures_csv_path:
    :return:
    """
    df = load_atlas_structures_csv(structures_csv_path)
    atlas_ids = get_all_structure_children(df, atlas_id)
    atlas = brainio.load_any(atlas_path)
    for idx in atlas_ids:
        region_mask = atlas == idx
        smoothed_region = smooth_structure(
            region_mask, threshold=smooth_threshold, sigma=smooth_sigma
        )
        brainrender.volume_to_vector_array_to_obj_file(
            smoothed_region, f"{out_dir}/{idx}.obj"
        )


def get_region_mask(atlas_id, atlas_path, structures_csv_path, smooth=False):
    df = load_atlas_structures_csv(structures_csv_path)
    atlas_ids = get_all_structure_children(df, atlas_id)
    all_regions = get_region(atlas_ids, atlas_path, smooth=smooth)
    return all_regions


def get_region(atlas_ids, atlas_path, smooth=True):
    atlas = brainio.load_any(atlas_path)
    all_regions = np.zeros_like(atlas)
    for id in atlas_ids:
        region_mask = atlas == id
        all_regions = np.logical_or(region_mask, all_regions)
    if smooth:
        all_regions = smooth_structure(all_regions)
    return all_regions


def get_arbitrary_structure_mask_from_custom_atlas(
    atlas_ids, atlas_path, sigma, smoothing_threshold
):
    """
    uses id numbers from custom atlas to generate a smoothed mask of those ids combined

    :param atlas_ids:
    :param atlas_path:
    :param sigma:
    :param smoothing_threshold:
    :return:
    """
    atlas = brainio.load_any(atlas_path)
    all_regions = np.zeros_like(atlas)
    for id in atlas_ids:
        region_mask = atlas == id
        all_regions = np.logical_or(region_mask, all_regions)
    all_regions = smooth_structure(
        all_regions, threshold=smoothing_threshold, sigma=sigma
    )
    return all_regions


def smooth_structure(image, threshold=0.4, sigma=10):
    for i in range(image.shape[1]):
        image[:, i, :] = gaussian(image[:, i, :], sigma) > threshold
    return image


def get_n_pixels_in_region(atlas_ids, atlas_path):
    all_regions_smooth = get_region(atlas_ids, atlas_path)
    return np.count_nonzero(all_regions_smooth)
