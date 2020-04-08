import napari
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from natsort import natsorted
from pathlib import Path
from imlib.general.system import get_text_lines
from imlib.IO.structures import load_structures_as_df
from imlib.anatomy.structures.structures_tree import (
    atlas_value_to_name,
    UnknownAtlasValue,
)
from neuro.atlas_tools.paths import Paths
from imlib.source.source_files import get_structures_path
from neuro.visualise.vis_tools import display_raw, display_downsampled, display_registration


def parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = cli_parse(parser)
    return parser


def cli_parse(parser):
    cli_parser = parser.add_argument_group("Visualisation options")

    cli_parser.add_argument(
        dest="amap_directory",
        type=str,
        help="Path to the amap output directory..",
    )

    cli_parser.add_argument(
        "-r" "--raw",
        dest="raw",
        action="store_true",
        help="Display the raw image (as a virtual stack) "
        "rather than the downsampled",
    )
    cli_parser.add_argument(
        "-c",
        "--raw-channels",
        dest="raw_channels",
        type=str,
        nargs="+",
        help="Paths to N additional raw channels to view. Will only work if "
        "using the raw image viewer.",
    )
    cli_parser.add_argument(
        "-m" "--memory",
        dest="memory",
        action="store_true",
        help="Load data into RAM. ",
    )

    return parser


def get_most_recent_log(directory, log_pattern="amap*.log"):
    """
    Returns the most recent amap log file (for parsing of arguments)
    :param directory:
    :param log_pattern: String pattern that defines the log
    :return: Path to the most recent log file
    """
    directory = Path(directory)
    return natsorted(directory.glob(log_pattern))[-1]


def read_log_file(
    log_file,
    log_entries_to_get=[
        "x_pixel_um",
        "y_pixel_um",
        "z_pixel_um",
        "image_paths",
        "registration_config",
    ],
    separator=": ",
):
    """
    Reads an amap log file, and returns a dict of entries corresponding to
    "log_entries_to_get"
    :param log_file: Path to the log file
    :param log_entries_to_get: List of strings corresponding to entries
    in the log file
    :param separator: Separator between the log item label and the entry.
    Default: ": "
    :return: A dict of the entries and labels
    """
    lines = get_text_lines(log_file)
    entries = {}
    for line in lines:
        for entry in log_entries_to_get:
            if line.startswith(entry):
                entries[entry] = line.strip(entry + separator)

    return entries


def main():
    print("Starting amap viewer")
    args = parser().parse_args()

    structures_path = get_structures_path()
    structures_df = load_structures_as_df(structures_path)

    if not args.memory:
        print(
            "By default amap_vis does not load data into memory. "
            "To speed up visualisation, use the '-m' flag. Be aware "
            "this will make the viewer slower to open initially."
        )

    paths = Paths(args.amap_directory)
    with napari.gui_qt():
        v = napari.Viewer(title="amap viewer")
        if (
            Path(paths.registered_atlas_path).exists()
            and Path(paths.boundaries_file_path).exists()
        ):

            if args.raw:
                image_scales = display_raw(v, args)

            else:
                if Path(paths.downsampled_brain_path).exists():
                    image_scales = display_downsampled(v, args, paths)

                else:
                    raise FileNotFoundError(
                        f"The downsampled image: "
                        f"{paths.downsampled_brain_path} could not be found. "
                        f"Please ensure this is the correct "
                        f"directory and that amap has completed. "
                    )

            labels = display_registration(
                v,
                paths.registered_atlas_path,
                paths.boundaries_file_path,
                image_scales,
                memory=args.memory,
            )

            @labels.mouse_move_callbacks.append
            def get_connected_component_shape(layer, event):
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

        else:
            raise FileNotFoundError(
                f"The directory: '{args.amap_directory}' does not "
                f"appear to be complete. Please ensure this is the correct "
                f"directory and that amap has completed."
            )


if __name__ == "__main__":
    main()
