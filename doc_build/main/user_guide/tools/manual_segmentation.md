# Manual segmentation

To manually segment brain regions in standard space that can then be visualised
along with other samples (e.g. in 
[BrainRender](https://github.com/BrancoLab/BrainRender).)


**N.B. This tool depends (for now) on 
[amap](https://github.com/SainsburyWellcomeCentre/amap). Please 
run `pip install amap` and then `amap_download` before running this tool if 
you don't already have cellfinder installed**

### Prerequisites
Data must be registered to a standard atlas (currently only the 
[Allen Reference Atlas](http://mouse.brain-map.org/) is supported) using 
[amap](https://github.com/SainsburyWellcomeCentre/amap-python) (or the amap 
based registration in 
[cellfinder](https://github.com/SainsburyWellcomeCentre/cellfinder)). Please 
follow the instructions for these packages, and ensure that the channel that 
you want to segment is downsampled (e.g. using the `--downsample` flag in 
amap).


## Usage
### Command line
```bash
    manual_region_seg "name_of_downsampled_image.nii" registration_directory
```

#### Arguments
Run `manual_region_seg -h` to see all options.

##### Positional arguments
* Downsampled image to be segmented, as a string (e.g. `"downsampled.nii"`)
* amap/cellfinder registration directory (e.g. 
`/home/analysis/cellfinder_output/registration/`)

##### The following options may also need to be used:
* `--save-image` Store the resulting segmented region image (e.g. for
inspecting in 2D. (default: False)
* `--preview` Preview the segmented regions in brainrender (default:False)
* `--debug` Debug mode. Will increase verbosity of logging and save all 
intermediate files for diagnosis of software issues. (default: False)


### napari GUI
manual_region_seg will transform your image into standard space (this may 
take a few minutes) and then display the image in a 
[napari](https://github.com/napari/napari) viewer.

