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
    region_parser = parser.add_argument_group("Region segmentation options")
    region_parser.add_argument(
        "--brush-size",
        dest="brush_size",
        type=int,
        default=30,
        help="Default size of the label brush.",
    )

    path_parser = parser.add_argument_group("Path segmentation options")
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
        "--regions",
        dest="regions",
        default=[],
        nargs="+",
        help="Brain regions to render, as acronyms. e.g. 'VISp MOp1'",
    )

    return parser
