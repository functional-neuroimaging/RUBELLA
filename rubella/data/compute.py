import numpy as np
from rabies.analysis_pkg.analysis_math import dual_regression


def vcorrcoef(X, Y):  # return a correlation between each row of X with each row of Y
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = (X-Xm).dot((Y-Ym).T)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1).reshape(-1,1)*np.sum((Y-Ym)**2, axis=1).reshape(1,-1))
    r = r_num/r_den
    return r


def compute_dual_regression(data_dict, prior_bold_idx):
    for fname in data_dict['fname_list']:
        timeseries = data_dict[fname]['timeseries']
        DR = dual_regression(data_dict['ICs_arr'], timeseries)
        DR_maps = DR['C']*DR['S']
        data_dict[fname]['DR_maps']=DR_maps[:,prior_bold_idx]
        data_dict[fname]['DR_amp']=DR['S'][prior_bold_idx]
        data_dict[fname]['DR_cross_corr_r']=np.corrcoef(DR['W'][:,prior_bold_idx].T)
        data_dict[fname]['DR_cross_corr_z']=np.arctanh(data_dict[fname]['DR_cross_corr_r'])


def compute_global_connectivity(data_dict, verbose=True):
    for fname in data_dict['fname_list']:
        if verbose:
            print(fname)
        data_dict[fname]['global_conn'] = np.corrcoef(data_dict[fname]['timeseries'].T).mean(axis=0)


def get_concat_timeseries(data_dict):
    fname_list = data_dict['fname_list']
    concat_timeseries = data_dict[fname_list[0]]['timeseries']
    for fname in fname_list[1:]:
        concat_timeseries = np.concatenate((concat_timeseries,data_dict[fname]['timeseries']),axis=0)
    print(concat_timeseries.shape)
    return concat_timeseries


def get_sliding_windows(data_dict, window_length, step_size=1, skip_censored_windows=True):
    # this returns a set of sliding windows concatenated across all data
    # set skip_censored_windows to True to exclude any window with censored data points, enforcing that each window is contiguous and the same length

    if step_size<1:
        raise ValueError(f"Step size must be at least 1, got {step_size} instead.")
    
    fname_list = data_dict['fname_list']
    for fname in fname_list:
        timeseries = data_dict[fname]['timeseries']
        if 'censoring_mask' in data_dict[fname].keys():
            censoring_mask = data_dict[fname]['censoring_mask']
            nframes = len(censoring_mask)
            idx_arr = np.array([None]*nframes)
            idx_arr[censoring_mask] = np.array(range(timeseries.shape[0]))
            censoring=True
        else:
            print(f'No censoring mask for {fname}')
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
        data_dict[fname]['sliding_windows']= windows
        data_dict[fname]['sliding_windows_idx']= window_idx_l # also save the timeseries index from which the windows were created
