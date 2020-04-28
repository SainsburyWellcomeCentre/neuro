from imlib.general.config import get_config_obj
from brainio import brainio
import json
from pathlib import Path


class Atlas(dict):
    def __init__(self, path):
        path = Path(path)

        # To temporarily support the old syntax, a conf file can still be passed.
        # If a conf file is used, we assume only one atlas is logged there - as currently
        # this seems to be the only supported option_
        # If we are already using new system:
        if path.is_dir():
            with open(path / "atlas_metadata.json", "r") as f:
                atlas_metadata = json.load(f)

            # This field we want to generate actively from instantiation path, to avoid confusion:
            self.base_folder = path
        else:
            # If a configuration file is passed:
            data_dict = dict(get_config_obj(str(path)))
            atlas_id = list(data_dict.keys())[0]  # assume only one key

            # Generate clean atlas metadata structure:
            atlas_metadata = data_dict[atlas_id]

            # Atlas name:
            atlas_metadata["atlas_id"] = atlas_id
            # Atlas base folder:
            self.base_folder = Path(atlas_metadata.pop("base_folder"))

            # To actively propagate support for new syntax, we also write the
            # json file in the atlas directory; this should discontinued in the long term:
            with open(self.base_folder / "atlas_metadata.json", "w") as f:
                json.dump(atlas_metadata, f)

        self._pix_sizes = None

        super().__init__(**atlas_metadata)

    @property
    def pix_sizes(self):
        # TODO can probably go safely away, or refactored in a more general inference method
        """ Get the dictionary of x, y, z from the after loading it
        or if the atlas size is default, use the values from the config file

        :return: The dictionary of x, y, z pixel sizes
        """
        if self._pix_sizes is None:
            pixel_sizes = self.get_nii_from_element(
                "atlas_name"
            ).header.get_zooms()
            if pixel_sizes != (0, 0, 0):
                self._pix_sizes = {
                    axis: round(size * 1000, 3)  # convert to um
                    for axis, size in zip(("x", "y", "z"), pixel_sizes)
                }
            else:
                self._pix_sizes = self["pixel_size"]
        return self._pix_sizes

    def get_element_path(self, element_name):
        """Get the path to an 'element' of the atlas (i.e. the average brain,
        the atlas, or the hemispheres atlas)

        :param str element_name: The name of the item to retrieve
        :return: The path to that atlas element on the filesystem
        :rtype: str
        """

        return self.base_folder / self[element_name]

    def get_left_hemisphere_value(self):
        return int(self["left_hemisphere_value"])

    def get_right_hemisphere_value(self):
        return int(self["right_hemisphere_value"])

    def get_nii_from_element(self, element_name):
        """ This can be easily changed to a different loading API if needed.
        """
        data_full_path = self.base_folder / self[element_name]
        return brainio.load_nii(data_full_path)
