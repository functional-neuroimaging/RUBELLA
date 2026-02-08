import numpy as np
from ..data.compute import vcorrcoef


def compute_QPP(data_dict, window_length, template_to_window_thresh = 0.1, max_iter=20, starting_idx='random'):
    '''
    Automated detection of a quasi-periodic pattern (QPP) as described in: 
    Majeed, W., ..., & Keilholz, S. D. (2011). Spatiotemporal dynamics of low frequency BOLD fluctuations in rats 
    and humans. Neuroimage, 54(2), 1140-1150.

    1. An initial QPP template is defined as one time window, selected randomly from the set of all sliding windows across the
    entire dataset, i.e. all data loaded in data_dict.
    2. The template is correlated with each sliding window from the entire dataset (i.e. a spatiotemporal correlation computed  
    across all voxels and timeframes in a window), and windows with a correlation above template_to_window_thresh are grouped 
    and then averaged to form an updated template.
    3. Repeat 2. until the previous template and the updated template are mutually correlated above 0.999, after which the
    updated template defines the finalized QPP.

    This function assumes that rubella.data.compute.get_sliding_windows was executed on data_dict a priori, with 
    skip_censored_windows=True to ensure contiguous windows.
    '''
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
        print(f'{num_idx} windows are averaged for the next template.')
        
        # average set of windows that pass corr. threshold to form a new template
        new_template = new_template.mean(axis=0).reshape(1,-1)
        
        template_corr = np.corrcoef(template,new_template)[0,1]
        template = new_template
        # repeat until convergence or X iterations
        if template_corr>0.999:
            converge=True
        if converge:
            break
    print(f'Converged after {i} cycles.')
    
    QPP = template.reshape(window_length,-1)
    return QPP


def count_QPP_occurence(data_dict, QPP, template_to_window_thresh = 0.2):
    '''
    Count QPP occurence within each scan as the number of windows that pass a correlation 
    threshold relative to the QPP template, as described in:
    Belloy, M. E., ... & Verhoye, M. (2018). Dynamic resting state fMRI analysis in mice reveals 
    a set of Quasi-Periodic Patterns and illustrates their relationship with the global signal. 
    Neuroimage, 180, 463-484.
    '''
    fname_list = data_dict['fname_list']
    template = QPP.reshape(1,-1)
    for fname in fname_list:
        windows = np.array(data_dict[fname]['sliding_windows'])
        windows = windows.reshape(windows.shape[0],-1) # need to reshape to 2D
        corrs=vcorrcoef(template, windows).flatten()
        idx = corrs>template_to_window_thresh # windows that pass threshold
        data_dict[fname]['QPP_counts'] = idx.sum()
    