import napari
from neuro.segmentation.manual_segmentation.widgets import General


def main(
    num_colors=10,
    brush_size=30,
    point_size=30,
    spline_size=10,
    track_file_extension=".h5",
    image_file_extension=".nii",
    x_scaling=10,
    y_scaling=10,
    z_scaling=10,
    spline_points_default=1000,
    spline_smoothing_default=0.1,
    fit_degree_default=3,
    summarise_track_default=True,
    add_surface_point_default=False,
    calculate_volumes_default=True,
    summarise_volumes_default=True,
    region_alpha_default=0.8,
    structure_alpha_default=0.8,
    shading_default="flat",
    region_to_add_default="",
):

    print("Loading manual segmentation GUI.\n ")
    with napari.gui_qt():

        viewer = napari.Viewer(title="Manual segmentation")
        general = General(
            viewer,
            point_size=point_size,
            spline_size=spline_size,
            x_scaling=x_scaling,
            y_scaling=y_scaling,
            z_scaling=z_scaling,
            track_file_extension=track_file_extension,
            image_file_extension=image_file_extension,
            brush_size=brush_size,
            num_colors=num_colors,
            spline_points_default=spline_points_default,
            spline_smoothing_default=spline_smoothing_default,
            fit_degree_default=fit_degree_default,
            summarise_track_default=summarise_track_default,
            add_surface_point_default=add_surface_point_default,
            calculate_volumes_default=calculate_volumes_default,
            summarise_volumes_default=summarise_volumes_default,
            region_alpha_default=region_alpha_default,
            structure_alpha_default=structure_alpha_default,
            shading_default=shading_default,
            region_to_add_default=region_to_add_default,
        )
        viewer.window.add_dock_widget(general, name="General", area="right")


if __name__ == "__main__":
    main()
