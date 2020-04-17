import argparse


def extraction_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="img_filepath", type=str, help="Path to brain volume (.nii) data",
    )

    parser.add_argument(
        dest="registration_folder",
        type=str,
        help="Path to cellfinder registration folder",
    )

    parser.add_argument(
        "-od",
        "--output-directory",
        dest="output_directory",
        type=str,
        default=None,
        help="Path to directory where the log will be saved.",
    )

    parser.add_argument(
        "-ow",
        "--overwrite",
        dest="overwrite",
        type=bool,
        default=True,
        help="If false files will not be overwritten.",
    )

    parser.add_argument(
        "-o",
        "--obj-path",
        dest="obj_path",
        type=str,
        default=None,
        help="Path to output .obj file. Will default to the image directory.",
    )

    parser.add_argument(
        "-k",
        "--gaussian-kernel",
        dest="gaussian_kernel",
        type=float,
        default=2,
        help="Float, size of kernel for gaussian smoothing (x,y directions).",
    )
    parser.add_argument(
        "--gaussian-kernel-z",
        dest="gaussian_kernel_z",
        type=float,
        default=6,
        help="Float, size of kernel for gaussian smoothing (zdirections).",
    )

    parser.add_argument(
        "-pt",
        "--percentile_threshold",
        dest="percentile_threshold",
        type=float,
        default=99.995,
        help="Float in range [0, 100]. The percentile number of pixel "
        "intensity values for tresholding",
    )

    parser.add_argument(
        "-tt",
        "--treshold-type",
        dest="threshold_type",
        type=str,
        default="otsu",
        help="'otsu' or 'percentile'. Determines how the threshold "
        "value is computed",
    )

    parser.add_argument(
        "-or",
        "--overwrite-registration",
        dest="overwrite_registration",
        type=str,
        default="False",
        help="If false skip running again the registration",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Debug mode. Will increase verbosity of logging and save all "
        "intermediate files for diagnosis of software issues.",
    )
    parser.add_argument(
        "--save-log",
        dest="save_log",
        action="store_true",
        help="Save logging to file (in addition to logging to terminal).",
    )
    return parser
