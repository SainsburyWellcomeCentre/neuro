from imlib.general.config import get_config_obj


def get_atlas_pixel_sizes(atlas_config_path):
    config_obj = get_config_obj(atlas_config_path)
    atlas_conf = config_obj["atlas"]
    atlas_pixel_sizes = atlas_conf["pixel_size"]
    return atlas_pixel_sizes
