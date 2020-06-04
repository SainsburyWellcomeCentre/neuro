from neuro.structures.structures_tree import (
    atlas_value_to_name,
    UnknownAtlasValue,
)


def display_brain_region_name(layer, structures_df):
    val = layer.get_value()
    if val != 0 and val is not None:
        try:
            region = atlas_value_to_name(val, structures_df)
            msg = f"{region}"
        except UnknownAtlasValue:
            msg = "Unknown region"
    else:
        msg = "No label here!"
    layer.help = msg
