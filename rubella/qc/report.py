import numpy as np
import matplotlib.pyplot as plt
from .. import plotting as plotting
from rabies.analysis_pkg.analysis_math import elementwise_corrcoef
from .utils import get_group_stats
from ..utils import _read_sitk_image

def group_stats_QC_plot(
        FC_map_dict, 
        brain_mask, 
        template_file,
        slice_selection=6, 
        slices_per_row=None, 
        plane='coronal',
        threshold=True,
        plot_scale=1.0, 
        vmax_FC=0.5, 
        top_percent=10,
        include_stat=['tstat', 'avg', 'sd', 'overlap'],
        qc_data_dict={},
        ):
    tstat_data, avg_FC, std_FC, percent_overlap = get_group_stats(FC_map_dict, avg_type='median', top_percent=top_percent)

    qc_keys = list(qc_data_dict.keys())
    num_maps = len(include_stat)+len(qc_keys)

    brain_mask = _read_sitk_image(brain_mask) # read once, instead of for each plot
    selected_slices = plotting.select_slices_for_plotting(slice_selection, brain_mask, plane)
    n_slices = len(selected_slices)
    if not slices_per_row:
        slices_per_row = n_slices # all slices in 1 row
    nrows_map = int(np.ceil(n_slices/slices_per_row))

    fig,axes = plt.subplots(num_maps,1, figsize=(slices_per_row*plot_scale,num_maps*nrows_map*plot_scale), constrained_layout=True)
    if num_maps==1:
        axes = [axes] # convert to list to avoid indexing error
    n=0
    if 'tstat' in include_stat:
        ax = axes[n]
        fixed_threshold = 2 if threshold else None
        vmax=20
        fig,ax,cbar = plotting.plot_brain(
                tstat_data,brain_mask,template_file,
                fig=fig,ax=ax,
                plane=plane, slice_selection=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='cold_hot', vmax=vmax, vmin=None, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold, # image thresholding
                )
        cbar.set_label(f't-value', rotation = 270, labelpad=10)
        cbar.set_ticks([-vmax,-fixed_threshold,fixed_threshold,vmax] if threshold else [-vmax,0,vmax])
        n+=1


    if 'avg' in include_stat:
        ax = axes[n]
        vmax=vmax_FC
        fixed_threshold = vmax*0.1 if threshold else None

        fig,ax,cbar = plotting.plot_brain(
                avg_FC,brain_mask,template_file,
                fig=fig,ax=ax,
                plane=plane, slice_selection=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='cold_hot', vmax=vmax, vmin=None, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold, # image thresholding
                )
        cbar.set_label(f'Dataset median', rotation = 270, labelpad=10)
        cbar.set_ticks([-vmax,-fixed_threshold,fixed_threshold,vmax] if threshold else [-vmax,0,vmax])
        n+=1

    if 'sd' in include_stat:
        ax = axes[n]

        fig,ax,cbar = plotting.plot_brain(
                std_FC,brain_mask,template_file,
                fig=fig,ax=ax,
                plane=plane, slice_selection=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='inferno', vmax=None, vmin=0, plot_cbar=True, # cmap range
                fixed_threshold=0, # image thresholding
                )
        cbar.set_label(f'Inter-scan variability', rotation = 270, labelpad=10)
        n+=1


    if 'overlap' in include_stat:
        ax = axes[n]
        vmax=100
        fixed_threshold = 10 if threshold else None

        fig,ax,cbar = plotting.plot_brain(
                percent_overlap,brain_mask,template_file,
                fig=fig,ax=ax,
                plane=plane, slice_selection=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='viridis', vmax=vmax, vmin=0, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold, # image thresholding
                )
        cbar.set_label(f'Percent overlap', rotation = 270, labelpad=10)
        cbar.set_ticks([fixed_threshold,vmax] if threshold else [0,vmax])
        n+=1

    qc_keys = list(qc_data_dict.keys())

    data_array = np.array([FC_map_dict[k] for k in FC_map_dict])
    for key in qc_keys:
        qc_corr_map = elementwise_corrcoef(data_array, np.array(qc_data_dict[key]).reshape(-1,1))

        ax = axes[n]
        vmax=0.5
        fixed_threshold = 0.1 if threshold else None

        fig,ax,cbar = plotting.plot_brain(
                qc_corr_map,brain_mask,template_file,
                fig=fig,ax=ax,
                plane=plane, slice_selection=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='cold_hot', vmax=vmax, vmin=-vmax, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold, # image thresholding
                )
        cbar.set_label(f'{key} corr.', rotation = 270, labelpad=10)
        cbar.set_ticks([-vmax,-fixed_threshold,fixed_threshold,vmax] if threshold else [-vmax,0,vmax])
        n+=1

    return fig,axes
    
