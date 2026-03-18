# The RUBELLA DataObject

```py
# first define a list of timeseries Nifti files in commonspace
import glob
timeseries_file_list = glob.glob('path_to_cleaned_nii_folder/*_cleaned.nii.gz')[:3]
timeseries_file_list.sort()
# if censoring was applied using RABIES, the list of censoring .csv files should be loaded too
censoring_file_list = glob.glob('path_to_censoring_csv_folder/*_frame_censoring_mask.csv')

from rubella.data_obj import DataObject
data_obj = DataObject(timeseries_file_list, brain_mask_file, img_id_list=None, fname_split='_cleaned', censoring_file_list=censoring_file_list, verbose=True)
```