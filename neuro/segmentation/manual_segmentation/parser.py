import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    main_parser = parser.add_argument_group("General options")
    main_parser.add_argument(
        dest="image",
        type=str,
        help="downsampled image to be segmented " "(as a string)",
    )
    main_parser.add_argument(
        dest="registration_directory",
        type=str,
        help="amap/cellfinder registration output directory",
    )
    main_parser.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="Preview the segmented regions in brainrender",
    )

    region_parser = parser.add_argument_group("Region segmentation options")
    region_parser.add_argument(
        "--volumes",
        dest="volumes",
        action="store_true",
        help="Calculate the volume of each brain area included in the "
        "segmented region",
    )
    region_parser.add_argument(
        "--summarise",
        dest="summarise",
        action="store_true",
        help="Summarise each region (centers, volumes etc.)",
    )
    region_parser.add_argument(
        "--shading",
        type=str,
        default="flat",
        help="Object shading type for brainrender ('flat', 'giroud' or "
        "'phong').",
    )
    region_parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Object transparency for brainrender.",
    )
    region_parser.add_argument(
        "--brush-size",
        dest="brush_size",
        type=int,
        default=30,
        help="Default size of the label brush.",
    )

    path_parser = parser.add_argument_group("Path segmentation options")
    path_parser.add_argument(
        "--surface-point",
        dest="add_surface_to_points",
        action="store_true",
        help="Find the closest part of the brain surface to the first point,"
        "and include that as a point for the spline fit.",
    )
    parser.add_argument(
        "--spline-radius",
        dest="spline_size",
        type=int,
        default=10,
        help="Radius of the visualised spline",
    )
    path_parser.add_argument(
        "--point-radius",
        dest="point_size",
        type=int,
        default=30,
        help="Radius of the visualised points",
    )
    path_parser.add_argument(
        "--region-alpha",
        dest="region_alpha",
        type=float,
        default=0.4,
        help="Brain region transparency for brainrender.",
    )
    path_parser.add_argument(
        "--regions",
        dest="regions",
        default=[],
        nargs="+",
        help="Brain regions to render, as acronyms. e.g. 'VISp MOp1'",
    )

    return parser
