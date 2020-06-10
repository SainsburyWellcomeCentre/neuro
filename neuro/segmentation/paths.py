from imlib.general.system import ensure_directory_exists

from neuro.atlas_tools import paths as reg_paths


class Paths:
    """
    A single class to hold all file paths that may be used. Any paths
    prefixed with "tmp__" refer to internal intermediate steps, and will be
    deleted if "--debug" is not used.
    """

    def __init__(self, registration_output_folder, downsampled_image):
        self.registration_output_folder = registration_output_folder

        self.segmentation_directory = (
            self.registration_output_folder / "manual_segmentation"
        )

        ensure_directory_exists(self.segmentation_directory)
        self.downsampled_image = self.join_seg_files(downsampled_image)

        self.tmp__inverse_transformed_image = self.join_seg_files(
            "image_standard_space.nii"
        )
        self.tmp__inverse_transform_log_path = self.join_seg_files(
            "inverse_transform_log.txt"
        )
        self.tmp__inverse_transform_error_path = self.join_seg_files(
            "inverse_transform_error.txt"
        )

        self.annotations = self.join_reg_files(reg_paths.ANNOTATIONS)
        self.hemispheres = self.join_reg_files(reg_paths.HEMISPHERES)

        self.regions_directory = self.join_seg_files("regions")
        self.region_summary_csv = self.regions_directory / "summary.csv"

        self.tracks_directory = self.join_seg_files("tracks")

    def join_reg_files(self, filename):
        return self.registration_output_folder / filename

    def join_seg_files(self, filename):
        return self.segmentation_directory / filename
