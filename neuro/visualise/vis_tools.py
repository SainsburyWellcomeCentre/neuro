from pathlib import Path
from natsort import natsorted

from imlib.general.config import get_config_obj
from imlib.general.system import get_text_lines


def get_image_scales(log_entries, config_file):
    """
    Returns the scaling from downsampled data to raw data
    :param log_entries: Entries parsed from the log file
    :param config_file: Path to the amap config file
    :return: Tuple of scaling factors
    """
    config_obj = get_config_obj(config_file)
    atlas_conf = config_obj["atlas"]
    pixel_sizes = atlas_conf["pixel_size"]
    x_scale = float(pixel_sizes["x"]) / float(log_entries["x_pixel_um"])
    y_scale = float(pixel_sizes["y"]) / float(log_entries["y_pixel_um"])
    z_scale = float(pixel_sizes["z"]) / float(log_entries["z_pixel_um"])
    return z_scale, y_scale, x_scale


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
            if line.startswith(entry + ":"):
                entries[entry] = line.strip(entry + separator)

    return entries
