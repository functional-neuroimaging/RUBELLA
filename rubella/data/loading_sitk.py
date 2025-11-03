import numpy as np
import SimpleITK as sitk
import pathlib
import pandas as pd

def load_commonspace_data_dict(timeseries_file_list, mask_file, IC_file=None):
    # all inputs are assumed to overlap in a commonspace
    data_dict = {}
    data_dict['mask_file'] = mask_file
    data_dict['IC_file'] = IC_file

    mask_img = sitk.ReadImage(mask_file)
    volume_idx = sitk.GetArrayFromImage(mask_img).astype(bool)
    data_dict['volume_idx'] = volume_idx
    if IC_file is not None:
        ICs_arr = sitk.GetArrayFromImage(sitk.ReadImage(IC_file))[:,volume_idx]
        data_dict['ICs_arr'] = ICs_arr

    fname_list = []
    for file in timeseries_file_list:
        fname = pathlib.Path(file).name
        if fname in fname_list:
            raise ValueError(f'Duplicate found for {fname}')
        fname_list.append(fname)
        print(file)
        data_dict[fname]={}
        timeseries = sitk.GetArrayFromImage(sitk.ReadImage(file))[:,volume_idx]
        data_dict[fname]['timeseries']=timeseries
    data_dict['fname_list'] = fname_list
    return data_dict


def load_censoring_mask(data_dict, censoring_file_list, fname_split):
    '''
    Here the censoring mask file is assigned to each timeseries fname based on matching filenames.
    fname_split is a string the will define how to split the timeseries fname and match the associated
    censoring mask filename.
    '''
    fname_list = data_dict['fname_list']
    for fname in fname_list:            
        # first remove any previous mask assignment
        if 'censoring_mask' in data_dict[fname].keys():
            print(f"Removing old censoring mask entry for {fname}")
            del data_dict[fname]['censoring_mask']

        scan_name = pathlib.Path(fname).name.rsplit(fname_split)[0]
        for censoring_file in censoring_file_list:
            if scan_name in censoring_file:
                if 'censoring_mask' in data_dict[fname].keys():
                    raise ValueError(f"{fname} is assigned to multiple censoring mask files - review the inputs and fname_split.")
                data_dict[fname]['censoring_mask']=pd.read_csv(censoring_file)['False = Masked Frames'].values

        if not 'censoring_mask' in data_dict[fname].keys():
            print(f'No censoring mask was assigned to {fname}.')    


def define_grouping(data_dict, group_str_list):
    for fname in data_dict['fname_list']:
        # first remove any previous group assignment
        if 'group' in data_dict[fname].keys():
            print(f"Removing old group entry {data_dict[fname]['group']} for {fname}")
            del data_dict[fname]['group']
        for group_str in group_str_list:
            if group_str in fname:
                if 'group' in data_dict[fname].keys():
                    raise ValueError(f"{fname} is assigned to multiple groups {data_dict[fname]['group']} and {group_str}.")
                data_dict[fname]['group']=group_str
        if not 'group' in data_dict[fname].keys():
            print(f'No group was assigned to {fname}.')
                
