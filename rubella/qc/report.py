import numpy as np
import matplotlib.pyplot as plt
from .. import plotting as plotting
from rabies.analysis_pkg.analysis_math import elementwise_corrcoef
from .utils import get_group_stats

def group_stats_QC_plot(
        FC_map_dict, 
        brain_mask_file, 
        template_file,
        selected_slices, 
        slices_per_row=8, 
        threshold=True, 
        vmax_FC=0.5, 
        include_stat=['tstat', 'avg', 'sd', 'overlap'],
        qc_data_dict={},
        ):
    tstat_data, avg_FC, std_FC, percent_overlap = get_group_stats(FC_map_dict, avg_type='median', top_percent=10)

    qc_keys = list(qc_data_dict.keys())
    num_maps = len(include_stat)+len(qc_keys)

    n_slices = len(selected_slices)
    nrows_map = int(np.ceil(n_slices/slices_per_row))

    fig,axes = plt.subplots(num_maps,1, figsize=(slices_per_row,num_maps*nrows_map), constrained_layout=True)
    if num_maps==1:
        axes = [axes] # convert to list to avoid indexing error
    n=0
    if 'tstat' in include_stat:
        ax = axes[n]
        fixed_threshold = 2 if threshold else None
        vmax=20
        fig,ax,cbar = plotting.plot_brain(
                tstat_data,brain_mask_file,template_file,
                fig=fig,ax=ax,
                plane='coronal', selected_slices=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='cold_hot', vmax=vmax, vmin=None, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold,percent_thresholding=False,top_percent=10, # image thresholding
                )
        cbar.set_label(f't-value', rotation = 270, labelpad=10)
        cbar.set_ticks([-vmax,-fixed_threshold,fixed_threshold,vmax])
        n+=1


    if 'avg' in include_stat:
        ax = axes[n]
        vmax=vmax_FC
        fixed_threshold = vmax*0.1 if threshold else None

        fig,ax,cbar = plotting.plot_brain(
                avg_FC,brain_mask_file,template_file,
                fig=fig,ax=ax,
                plane='coronal', selected_slices=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='cold_hot', vmax=vmax, vmin=None, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold,percent_thresholding=False,top_percent=10, # image thresholding
                )
        cbar.set_label(f'Median Correlation', rotation = 270, labelpad=10)
        cbar.set_ticks([-vmax,-fixed_threshold,fixed_threshold,vmax])
        n+=1

    if 'sd' in include_stat:
        ax = axes[n]

        fig,ax,cbar = plotting.plot_brain(
                std_FC,brain_mask_file,template_file,
                fig=fig,ax=ax,
                plane='coronal', selected_slices=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='inferno', vmax=None, vmin=0, plot_cbar=True, # cmap range
                fixed_threshold=0,percent_thresholding=False,top_percent=10, # image thresholding
                )
        cbar.set_label(f'Inter-scan variability', rotation = 270, labelpad=10)
        n+=1


    if 'overlap' in include_stat:
        ax = axes[n]
        vmax=100
        fixed_threshold = 10 if threshold else None

        fig,ax,cbar = plotting.plot_brain(
                percent_overlap,brain_mask_file,template_file,
                fig=fig,ax=ax,
                plane='coronal', selected_slices=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='viridis', vmax=vmax, vmin=0, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold,percent_thresholding=False,top_percent=10, # image thresholding
                )
        cbar.set_label(f'Percent overlap', rotation = 270, labelpad=10)
        cbar.set_ticks([fixed_threshold,vmax])
        n+=1

    qc_keys = list(qc_data_dict.keys())

    data_array = np.array([FC_map_dict[k] for k in FC_map_dict])
    for key in qc_keys:
        qc_corr_map = elementwise_corrcoef(data_array, np.array(qc_data_dict[key]).reshape(-1,1))

        ax = axes[n]
        vmax=0.5
        fixed_threshold = 0.1 if threshold else None

        fig,ax,cbar = plotting.plot_brain(
                qc_corr_map,brain_mask_file,template_file,
                fig=fig,ax=ax,
                plane='coronal', selected_slices=selected_slices, slices_per_row=slices_per_row, # slice selection
                cmap='cold_hot', vmax=vmax, vmin=-vmax, plot_cbar=True, # cmap range
                fixed_threshold=fixed_threshold,percent_thresholding=False,top_percent=10, # image thresholding
                )
        cbar.set_label(f'{key} corr.', rotation = 270, labelpad=10)
        cbar.set_ticks([-vmax,-fixed_threshold,fixed_threshold,vmax])
        n+=1

    return fig,axes
    
