# RUBELLA: Repository of Utilitaries for Brain Experiments Low Level Analyses.

## Documentation

The full documentation is available online at [rubella.readthedocs.io](https://rubella.readthedocs.io).
It is rebuilt automatically from the `main` branch, and documentation for a specific release can be selected from the version menu at the bottom of the sidebar.

## Recommended installation

Below is a template to create a conda environment with RUBELLA installed
```sh
conda install -n base -c conda-forge mamba # this installs the mamba package manager, which installs packages 
# much faster than using the native conda install

mamba env create -f rubella_env.yml -n rubella_env # the .yml file contains a list of python dependencies you
# need to run RUBELLA. They are listed with specific versions that we know work, so this is a more robust method
# to setup you conda environment
# `-n rubella_env` is the name of your environment, you can give it another name if you wish

conda activate rubella # activate your python environment, so that you will install rubella inside with the 
# next command

pip install -e . # this will install the rubella package, and it is ready for import when you code in python 
# using the newly created environment
```

## Quick introduction: data loading, computing functional connectivity, and visualizing outputs

RUBELLA begins by loading a list of timeseries Niftis files (that are preprocessed and overlap in a commonspace) using the DataObject class, which stores all the timeseries in a common array format, ready for computations.
```py

# provide a path to the commonspace brain mask
brain_mask_file = '/path_to_mask_folder/commonspace_template_brain_mask_resampled.nii.gz'

# define a list of timeseries Nifti files in commonspace
import glob
timeseries_file_list = glob.glob('path_to_cleaned_nii_folder/*_cleaned.nii.gz')
timeseries_file_list.sort()
# if censoring was applied using RABIES, the list of censoring .csv files should be loaded too
censoring_file_list = glob.glob('path_to_censoring_csv_folder/*_frame_censoring_mask.csv')

from rubella.data_obj import DataObject
data_obj = DataObject(brain_mask_file, timeseries_file_list, img_id_list=None, fname_split='_cleaned', censoring_file_list=censoring_file_list, verbose=True)
```
You should become familiar with RUBELLA's DataObject, as it is at the core of how the package handles computations at the dataset level, and the object stores the key input for downstream functions. You should its docstring by typing `help(DataObject)` in python or by consulting the reference API in the documentation.

Next, downstream analyses across all images can be conveniently computed using various implementations of
functional connectivity analyses in the package... for example for global connectivity:
```py
from rubella.basic_fc import compute_global_connectivity
GC_dict = compute_global_connectivity(data_obj.timeseries_dict, data_obj.img_id_list, verbose=True)
```
Most functional connectivity functions in RUBELLA stores the output in a python dictionary format, where
the connectivity measure for a given scan is accessed using its image ID within the DataObject. For instance,
`GC_dict` is a python dictionary that stores the computed global connectivity maps for each image, and you
can loop through all images and access its global connectivity with:
```
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