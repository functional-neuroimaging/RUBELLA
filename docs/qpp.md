# Quasi-periodic pattern analysis (QPP)

The `qpp/` folder regroups functions associated to QPP analysis. The main function implemented the automated QPP detection algorithm described in: 
Majeed, W., ..., & Keilholz, S. D. (2011). Spatiotemporal dynamics of low frequency BOLD fluctuations in rats 
and humans. Neuroimage, 54(2), 1140-1150.

1. An initial QPP template is defined as one time window, selected randomly from the set of all sliding windows across the
entire dataset, i.e. all data loaded in data_dict.
2. The template is correlated with each sliding window from the entire dataset (i.e. a spatiotemporal correlation computed  
across all voxels and timeframes in a window), and windows with a correlation above template_to_window_thresh are grouped 
and then averaged to form an updated template.
3. Repeat 2. until the previous template and the updated template are mutually correlated above 0.999, after which the
updated template defines the finalized QPP.


### First the data must be subdivided into sliding windows a priori
```python
from rubella.data import compute
window_length=10
compute.get_sliding_windows(data_dict, window_length=window_length, step_size=1, skip_censored_windows=True)
```

### Then a QPP can be generated as follows
```python
from rubella.qpp.main import compute_QPP
QPP = compute_QPP(data_dict, window_length=window_length, template_to_window_thresh = 0.1, max_iter=10, starting_idx='random')
```
The ideal value for template_to_window_thresh will likely depend on the spatial smoothness of the input data. Also to ensure deterministic outputs, a fixed integer value for starting_idx can be provided.

### ...and the occurence rate for each scan can be then evaluated by counting windows above template_to_window_thresh
```python
from rubella.qpp.main import count_QPP_occurence
count_QPP_occurence(data_dict, QPP, template_to_window_thresh = 0.1, remove_consecutives=False)
fname_list = data_dict['fname_list']
for fname in fname_list: # loop throughout the scans to inspect the newly computed QPP occurence count
    print(fname)
    print(data_dict[fname]['QPP_counts'])
```
Select `remove_consecutives=True` to avoid counting a single QPP cycle multiple times across neighboring sliding windows.