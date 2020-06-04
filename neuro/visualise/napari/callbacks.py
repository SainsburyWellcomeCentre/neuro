from neuro.structures.structures_tree import (
    atlas_value_to_name,
    UnknownAtlasValue,
)
from neuro.visualise.napari.layers import add_new_label_layer

from brainio.brainio import load_any
from imlib.general.system import (
    ensure_directory_exists,
    delete_directory_contents,
)
from neuro.segmentation.manual_segmentation.man_seg_tools import (
    save_regions_to_file,
    analyse_region_brain_areas,
    summarise_brain_regions,
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


def add_label_layer(
    viewer, label_layers, image_like, num_colors=10, brush_size=30
):
    label_layers.append(
        add_new_label_layer(
            viewer,
            image_like,
            name="new_region",
            brush_size=brush_size,
            num_colors=num_colors,
        )
    )


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
