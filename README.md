# RUBELLA: Repository of Utilitaries for Brain Experiments Low Level Analyses.

## Files

* `rubella/`          ...the software package
* `docs/`             ...detailed documentation of RUBELLA's usage
* `contributing.md`   ...instructions for committing new code
* `environment.yml`   ...a python environment file with all required dependencies
* `setup.py`          ...install the python package so it can be imported

## Python setup

```sh
# install miniconda/anaconda which handles python environment
# then you can install mamba, which is faster than conda for creating an environment, conda install can be very slow
conda install -n base -c conda-forge mamba
mamba env create -f rubella_env.yml # the .yml file will create a 'rubella' environment will all required dependencies
conda activate rubella # activate your python environment
pip install -e . # this will install the rubella package, which can then be imported in python
# you're set, now you can run python with the package installed
```

## Quick start
RUBELLA begins by loading a list of timeseries Niftis into a DataObject.
```py

# provide a path to the commonspace brain mask
brain_mask_file = '/path_to_mask_folder/commonspace_template_brain_mask_resampled.nii.gz'

# define a list of timeseries Nifti files in commonspace
import glob
timeseries_file_list = glob.glob('path_to_cleaned_nii_folder/*_cleaned.nii.gz')[:3]
timeseries_file_list.sort()
# if censoring was applied using RABIES, the list of censoring .csv files should be loaded too
censoring_file_list = glob.glob('path_to_censoring_csv_folder/*_frame_censoring_mask.csv')

from rubella.data_obj import DataObject
data_obj = DataObject(timeseries_file_list, brain_mask_file, img_id_list=None, fname_split='_cleaned', censoring_file_list=censoring_file_list, verbose=True)
```

Then downstream analyses across all images are conveniently computed... for example for global connectivity:
```py
from rubella.basic import compute_global_connectivity
GC_dict = compute_global_connectivity(data_obj.timeseries_dict, data_obj.img_id_list, verbose=True)
# GC_dict is a dictionary of global connectivity maps for each image loaded in the DataObject
# each entry in the dictionary is accessed from the list of image IDs, img_id_list
for img_id in data_obj.img_id_list:
    global_connectivity = GC_dict[img_id]
```

Then the resulting connectivity maps can be visualized using RUBELLA's plotting functions:
```py
from rubella import plotting
template_file = '/path_to_template_folder/commonspace_template.nii.gz'
brain_mask_file = data_obj.brain_mask_file

data = GC_dict[data_obj.img_id_list[0]] # inspecting the first image in the list

fig,ax_list,cbar_list = plotting.plot_brain(
        data,brain_mask_file,template_file,
        plane='coronal', selected_slices='all', slices_per_row=6,
        plot_cbar=True,
        )
```