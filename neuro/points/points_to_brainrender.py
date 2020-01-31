"""
Converts point positions from cellfinder coordinates to brainrender

N.B. This is currently specific to coronal images, with the origin at the most
caudal, ventral left point. The default is also for 10um voxel spacing.
"""

import argparse
import imlib.IO.cells as cells_io
from imlib.general.numerical import check_positive_float, check_positive_int


def run(
    cells_file,
    output_filename,
    pixel_size_x=10,
    pixel_size_y=10,
    pixel_size_z=10,
    max_z=13200,
    key="df",
):
    print(f"Converting file: {cells_file}")
    cells = cells_io.get_cells(cells_file)
    for cell in cells:
        cell.transform(
            x_scale=pixel_size_x,
            y_scale=pixel_size_y,
            z_scale=pixel_size_z,
            integer=True,
        )

    cells = cells_io.cells_to_dataframe(cells)
    cells.columns = ["z", "y", "x", "type"]

    cells["x"] = max_z - cells["x"]

    print(f"Saving to: {output_filename}")
    cells.to_hdf(output_filename, key=key, mode="w")

    print("Finished")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="cells_file",
        type=str,
        help="Cellfinder cells file to be converted",
    )
    parser.add_argument(
        dest="output_filename",
        type=str,
        help="Output filename. Should end with '.h5'",
    )

    parser.add_argument(
        "-x",
        "--x-pixel-size",
        dest="x_pixel_size",
        type=check_positive_float,
        default=10,
        help="Pixel_size that the cells are defined in.",
    )
    parser.add_argument(
        "-y",
        "--y-pixel-size",
        dest="y_pixel_size",
        type=check_positive_float,
        default=10,
        help="Pixel_size that the cells are defined in.",
    )
    parser.add_argument(
        "-z",
        "--z-pixel-size",
        dest="z_pixel_size",
        type=check_positive_float,
        default=10,
        help="Pixel_size that the cells are defined in.",
    )
    parser.add_argument(
        "--max-z",
        dest="max_z",
        type=check_positive_int,
        default=13200,
        help="Maximum z extent of the atlas",
    )
    parser.add_argument(
        "--hdf-key",
        dest="hdf_key",
        type=str,
        default="df",
        help="hdf identifier ",
    )
    return parser


def main():
    args = get_parser().parse_args()
    run(
        args.cells_file,
        args.output_filename,
        pixel_size_x=args.x_pixel_size,
        pixel_size_y=args.y_pixel_size,
        pixel_size_z=args.z_pixel_size,
        max_z=args.max_z,
        key=args.hdf_key,
    )


if __name__ == "__main__":
    main()
