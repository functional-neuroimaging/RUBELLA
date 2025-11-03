import numpy as np
from rabies.analysis_pkg.analysis_math import dual_regression
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


def get_sliding_windows(data_dict, length):
    # this returns a set of sliding windows concatenated across all data
    windows = []
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

        for i in range(nframes-length):
            if (not censoring) or (censoring_mask[i:i+length].sum()==length): # only compute a window if it is contiguous
                window_idx = idx_arr[i:i+length].astype(int)
                windows.append(timeseries[window_idx,:].flatten())
                
    windows = np.array(windows)
    return windows


def vcorrcoef(X, Y):  # return a correlation between each row of X with each row of Y
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = (X-Xm).dot((Y-Ym).T)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1).reshape(-1,1)*np.sum((Y-Ym)**2, axis=1).reshape(1,-1))
    r = r_num/r_den
    return r


def compute_QPP(windows, length, template_to_frame_thresh = 0.05, max_iter=20, starting_idx='random'):
    if starting_idx=='random':
        from random import randrange
        starting_idx = randrange(windows.shape[0])
    if not isinstance(starting_idx, int):
        raise ValueError("starting_idx must be an integer.")

    template = windows[[starting_idx],:]

    converge=False
    for i in range(max_iter):
        # compute corr. between each window and the template
        corrs = vcorrcoef(X=windows, Y=template).flatten()

        # average set of windows that pass corr. threshold to form a new template
        idx = corrs>template_to_frame_thresh
        print(idx.sum())
        new_template = windows[idx].mean(axis=0).reshape(1,-1)
        template_corr = np.corrcoef(template,new_template)[0,1]
        template = new_template
        # repeat until convergence or X iterations
        if template_corr>0.999:
            converge=True
        if converge:
            break
    print(i)

    QPP = template.reshape(length,-1)
    return QPP