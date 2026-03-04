import numpy as np
import SimpleITK as sitk
from rabies.analysis_pkg.analysis_math import dual_regression


def vcorrcoef(X, Y):  # return a correlation between each row of X with each row of Y
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = (X-Xm).dot((Y-Ym).T)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1).reshape(-1,1)*np.sum((Y-Ym)**2, axis=1).reshape(1,-1))
    r = r_num/r_den
    return r


def compute_dual_regression(data_obj, IC_file, IC_idx=None):
    ICs_arr = sitk.GetArrayFromImage(sitk.ReadImage(IC_file))[:,data_obj.volume_idx]

    if not IC_idx:
        IC_idx = list(range(ICs_arr.shape[0])) # take all ICs

    DR_dict = {}
    for img_id in data_obj.img_id_list:
        timeseries = data_obj.timeseries_dict[img_id]
        DR = dual_regression(ICs_arr, timeseries)
        DR_maps = DR['C']*DR['S']
        DR_dict[img_id] = {}
        DR_dict[img_id]['DR_maps']=DR_maps[:,IC_idx]
        DR_dict[img_id]['DR_amp']=DR['S'][IC_idx]
        DR_dict[img_id]['DR_cross_corr_r']=np.corrcoef(DR['W'][:,IC_idx].T)
        DR_dict[img_id]['DR_cross_corr_z']=np.arctanh(DR_dict[img_id]['DR_cross_corr_r'])
    return DR_dict


def compute_global_connectivity(data_obj, verbose=True):
    GC_dict = {}
    for img_id in data_obj.img_id_list:
        if verbose:
            print(img_id)
        cross_corr = np.corrcoef(data_obj.timeseries_dict[img_id].T)
        cross_corr[np.isnan(cross_corr)] = 0
        GC_dict[img_id] = cross_corr.mean(axis=0)
        del cross_corr
    return GC_dict


def get_concat_timeseries(data_obj):
    img_id_list = data_obj.img_id_list
    concat_timeseries = data_obj.timeseries_dict[img_id_list[0]]
    for img_id in img_id_list[1:]:
        concat_timeseries = np.concatenate((concat_timeseries,data_obj.timeseries_dict[img_id]),axis=0)
    print(concat_timeseries.shape)
    return concat_timeseries


def get_sliding_windows(data_obj, window_length, step_size=1, skip_censored_windows=True):
    # this returns a set of sliding windows concatenated across all data
    # set skip_censored_windows to True to exclude any window with censored data points, enforcing that each window is contiguous and the same length

    if step_size<1:
        raise ValueError(f"Step size must be at least 1, got {step_size} instead.")
    
    sliding_window_dict = {}
    img_id_list = data_obj.img_id_list
    for img_id in img_id_list:
        timeseries = data_obj.timeseries_dict[img_id]
        if hasattr(data_obj, "censoring_mask") and img_id in data_obj.censoring_mask.keys():
            censoring_mask = data_obj.censoring_mask[img_id]
            nframes = len(censoring_mask)
            idx_arr = np.array([None]*nframes)
            idx_arr[censoring_mask] = np.array(range(timeseries.shape[0]))
            censoring=True
        else:
            print(f'No censoring mask for {img_id}')
            nframes = timeseries.shape[0]
            idx_arr = np.array(range(timeseries.shape[0]))
            censoring=False

        windows = []
        window_idx_l = []
        i=0
        while i<(nframes-window_length):
            # include if skip_censored_windows=False, censoring was not applied, or no data points were censored
            window_censoring_mask = censoring_mask[i:i+window_length]
            if (not skip_censored_windows) or (not censoring) or (window_censoring_mask.sum()==window_length): # only compute a window if it is contiguous
                window_idx = idx_arr[i:i+window_length][window_censoring_mask].astype(int) # censoring is applied to remove the None idx from idx_arr
                windows.append(timeseries[window_idx,:])
                window_idx_l.append(window_idx)
            i+=step_size
        sliding_window_dict[img_id] = {}
        sliding_window_dict[img_id]['sliding_windows']= windows
        sliding_window_dict[img_id]['sliding_windows_idx']= window_idx_l # also save the timeseries index from which the windows were created
    return sliding_window_dict
