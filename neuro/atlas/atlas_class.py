from imlib.general.config import get_config_obj
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

        super().__init__(**atlas_metadata)