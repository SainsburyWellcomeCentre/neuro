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
* `--preview` Preview the segmented regions in brainrender (default:False)
* `--debug` Debug mode. Will increase verbosity of logging and save all 
intermediate files for diagnosis of software issues. (default: False)


### napari GUI
manual_region_seg will transform your image into standard space (this may 
take a few minutes) and then display the image in a 
[napari](https://github.com/napari/napari) viewer:

<img src="https://raw.githubusercontent.com/SainsburyWellcomeCentre/neuro/master/resources/manual_segmentation_window.png" alt="manual_seg_window" width="700"/>

##### To segment regions:
* Ensure that the "Regions" tab is selected (left hand side)
* Navigate to where you want to draw your region of interest.
    * Use the scroll bar at the bottom (or left/right keys) to navigate 
    through the image stack
    * Use the mouse scrollwheel to zoom in or out
    * Drag with the mouse the pan the view
    
* Select a label ID (by pressing the `+` button in the `label` row, top-left),
the ID is not important, but `0` refers to no label, so you may as well start 
from 1.
* Choose a brush size (also in top left box)
* Activate painting mode (by selecting the paintbrush, top left). You can 
go back to the navigation mode by selecting the magnifying glass.
* Colour in your region that you want to segment, ensuring that you make a 
solid object. 
* Selecting the `ndim` toggle in the top left will extend the brush size in 
three dimensions (so it will colour in multiple layers).

* Repeat above for each region you wish to segment.
* Press `Control+S` on your keyboard to save the regions. If you used the
 `--preview flag`, once they are saved, they will be displayed in a brainrender
 window.


##### Editing regions:
If you have already run `manual_region_seg`, and run it again, the segmented 
regions will be shown. You can edit them, and press `Control+S` to resave them. 
If you don't want to save any changes, press `Control+X` to exit. The regions
 will still be previewed if you have selected that option.