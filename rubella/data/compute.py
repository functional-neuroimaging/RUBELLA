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


def compute_QPP(data_dict, window_length, template_to_window_thresh = 0.05, max_iter=20, starting_idx='random'):
    fname_list = data_dict['fname_list']

    for fname in fname_list:
        for window in data_dict[fname]['sliding_windows']:
            if not window.shape[0]==window_length:
                raise ValueError(f"The function is expecting windows of length {window_length}, but a window of length {window.shape[0]} was found for scan id {fname}")

    num_scan = len(fname_list)
    num_windows_l = [len(data_dict[fname]['sliding_windows']) for fname in fname_list]
    num_windows_total = np.array(num_windows_l).sum()

    def match_window_idx(idx):
        i=0
        for n in range(num_scan):
            if i+num_windows_l[n]>idx:
                scan_idx = idx-i
                fname=fname_list[n]
                return data_dict[fname]['sliding_windows'][scan_idx]
            i+=num_windows_l[n]
        raise

    if starting_idx=='random':
        from random import randrange
        starting_idx = randrange(num_windows_total)
        print(f'Template initiated with random index {starting_idx}')
    if not isinstance(starting_idx, int):
        raise ValueError("starting_idx must be an integer.")

    template = match_window_idx(starting_idx).reshape(1,-1)

    converge=False
    for i in range(max_iter):
        # compute corr. between each window and the template
        num_idx = 0
        new_template = np.zeros((0,template.shape[1]))
        for fname in fname_list:
            windows = np.array(data_dict[fname]['sliding_windows'])
            windows = windows.reshape(windows.shape[0],-1) # need to reshape to 2D
            corrs=vcorrcoef(template, windows).flatten()
            idx = corrs>template_to_window_thresh # windows that pass threshold
            new_template = np.append(new_template, windows[idx], axis=0)
            num_idx+=idx.sum()
        print(num_idx)
        
        # average set of windows that pass corr. threshold to form a new template
        new_template = new_template.mean(axis=0).reshape(1,-1)
        
        template_corr = np.corrcoef(template,new_template)[0,1]
        template = new_template
        # repeat until convergence or X iterations
        if template_corr>0.999:
            converge=True
        if converge:
            break
    print(i)
    
    QPP = template.reshape(window_length,-1)
    return QPP


def count_QPP_occurence(data_dict, QPP, template_to_window_thresh = 0.2):
    fname_list = data_dict['fname_list']
    template = QPP.reshape(1,-1)
    for fname in fname_list:
        corrs=vcorrcoef(template, data_dict[fname]['sliding_windows']).flatten()
        idx = corrs>template_to_window_thresh # windows that pass threshold
        data_dict[fname]['QPP_counts'] = idx.sum()
    