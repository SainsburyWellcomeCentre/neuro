from brainio.brainio import load_any
from imlib.general.system import (
    ensure_directory_exists,
    delete_directory_contents,
)
from neuro.segmentation.manual_segmentation.man_seg_tools import (
    convert_and_save_points,
)
from neuro.segmentation.manual_segmentation.man_seg_tools import (
    save_regions_to_file,
    analyse_region_brain_areas,
    summarise_brain_regions,
    analyse_track,
    analyse_track_anatomy,
)
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


def track_analysis(
    viewer,
    scene,
    track_points_file,
    points_layers,
    x_scaling,
    y_scaling,
    z_scaling,
    summary_csv_file=None,
    add_surface_to_points=True,
    spline_points=100,
    fit_degree=3,
    spline_smoothing=0.05,
    point_size=30,
    spline_size=10,
    summarise_track=True,
):
    max_z = len(viewer.layers[0].data)
    convert_and_save_points(
        points_layers,
        track_points_file,
        x_scaling,
        y_scaling,
        z_scaling,
        max_z,
    )

    scene, spline = analyse_track(
        scene,
        track_points_file,
        add_surface_to_points=add_surface_to_points,
        spline_points=spline_points,
        fit_degree=fit_degree,
        spline_smoothing=spline_smoothing,
        point_radius=point_size,
        spline_radius=spline_size,
    )
    if summarise_track:
        if summary_csv_file:
            analyse_track_anatomy(scene, spline, summary_csv_file)

    return scene, spline


def region_analysis(
    label_layers,
    structures_df,
    regions_directory,
    annotations_path,
    hemispheres_path,
    image_like,
    output_csv_file=None,
    volumes=True,
    summarise=True,
):
    ensure_directory_exists(regions_directory)
    delete_directory_contents(str(regions_directory))
    if volumes:
        print("Calculating region volume distribution")
        annotations = load_any(annotations_path)
        hemispheres = load_any(hemispheres_path)

        print(f"\nSaving summary volumes to: {regions_directory}")
        for label_layer in label_layers:
            analyse_region_brain_areas(
                label_layer,
                regions_directory,
                annotations,
                hemispheres,
                structures_df,
            )
    if summarise:
        if output_csv_file is not None:
            print("Summarising regions")
            summarise_brain_regions(label_layers, output_csv_file)

    print(f"\nSaving regions to: {regions_directory}")
    for label_layer in label_layers:
        save_regions_to_file(
            label_layer, regions_directory, image_like,
        )
