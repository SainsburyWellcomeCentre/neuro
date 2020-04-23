from imlib.general.config import get_config_obj


def get_atlas_pixel_sizes(atlas_config_path):
    config_obj = get_config_obj(atlas_config_path)
    atlas_conf = config_obj["atlas"]
    atlas_pixel_sizes = atlas_conf["pixel_size"]
    return atlas_pixel_sizes


def get_voxel_volume(registration_config):
    config_obj = get_config_obj(registration_config)
    atlas_conf = config_obj["atlas"]
    atlas_pixel_sizes = atlas_conf["pixel_size"]
    x_pixel_size = float(atlas_pixel_sizes["x"])
    y_pixel_size = float(atlas_pixel_sizes["y"])
    z_pixel_size = float(atlas_pixel_sizes["z"])

    voxel_volume = x_pixel_size * y_pixel_size * z_pixel_size
    return voxel_volume
