def lateralise_atlas(
    atlas, hemispheres, left_hemisphere_value=2, right_hemisphere_value=1
):
    atlas_left = atlas[hemispheres == left_hemisphere_value]
    atlas_right = atlas[hemispheres == right_hemisphere_value]
    return atlas_left, atlas_right
