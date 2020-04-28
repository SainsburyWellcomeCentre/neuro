# Heatmap generation

To generate a heatmap of detected cells, which shows cell distributions in a 
more intuitive way that showing individual cell positions:


<img src="https://raw.githubusercontent.com/SainsburyWellcomeCentre/cellfinder/master/resources/heatmap.png" alt="heatmap" width="500"/>

*Overlay on raw data and segentation from 
[amap](https://github.com/SainsburyWellcomeCentre) added separately*


### Usage
```bash
    heatmap cell_classification.xml heatmap.nii raw_data registered_atlas.nii -x 2 -y 2 -z 5
```

### Arguments
Run `heatmap -h` to see all options.

#### Positional arguments
* [Cellfinder](https://github.com/SainsburyWellcomeCentre/cellfinder) 
classified cells file (usually `cell_classification.xml`)
* Output filename. Should end with '.nii'. If the containing directory doesn't 
exist, it will be created.
* Path to raw data (just a single channel). Used to find the shape of the 
raw image that the detected cell positions are defined in.
* Registered atlas file from [amap](https://github.com/SainsburyWellcomeCentre)
 (typically run automatically in 
 [Cellfinder](https://github.com/SainsburyWellcomeCentre/cellfinder)). File 
 is usually `registered_atlas.nii`.
 

#### Keyword arguments
* `-x` or `--x-pixel-size` Pixel spacing of the data that the cells are 
defined in, in the first dimension, specified in um.
* `-y` or `--y-pixel-size` Pixel spacing of the data that the cells are 
defined in, in the second dimension, specified in um.
* `-z` or `--z-pixel-size` Pixel spacing of the data that the cells are 
defined in, in the third dimension, specified in um. 


#### The following options may also need to be used:
* `--heatmap-bin` Heatmap bin size (um of each edge of histogram cube)
* `--heatmap-smoothing` Gaussian smoothing sigma, in um.
* `--no-mask-figs` Don't mask the figures (removing any areas outside the 
brain, from e.g. smoothing)
