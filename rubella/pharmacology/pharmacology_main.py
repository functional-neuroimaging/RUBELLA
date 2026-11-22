import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nb
import matplotlib.patches as mpatches
import SimpleITK as sitk
from nilearn.plotting import plot_stat_map 
import numpy as np
import matplotlib.pyplot as plt
import os

import rubella.data_obj
import rubella.basic
import rubella.plotting

def compute_percent_change_from_baseline(data_obj, output_dir: str, reference_nifti: str,
                                         inj_time, TR, total_time):
    '''
    Compute PSC and save output as nifti.

    If the scan duration equals total_time, censoring is applied first to remove
    motion spikes, then uncensored to replace them with NaNs.
    If the scan is already shorter than total_time, it has already been censored
    and we skip straight to uncensoring.

    Parameters
    ----------
    data_obj        : DataObject with .timeseries_masked_dict, .censoring_mask, .volume_idx
    output_dir      : directory where output niftis will be saved
    reference_nifti : path to a 3D nifti for sitk spatial metadata
    inj_time        : injection time in minutes
    TR              : repetition time in seconds
    total_time      : total expected scan duration in minutes
    '''

    os.makedirs(output_dir, exist_ok=True)

    mask_arr_sitk  = data_obj.volume_idx
    ref_sitk       = sitk.ReadImage(reference_nifti)
    n_tp_expected  = int((total_time * 60) / TR)

    inj_timepoint  = int((inj_time * 60) / TR)
    baseline_start = int(2 * (60 / TR))
    baseline_end   = inj_timepoint
    print(f"Baseline window: timepoints {baseline_start} to {baseline_end}")

    for key in data_obj.censoring_mask.keys():

        censor_mask = data_obj.censoring_mask[key]
        n_tp_actual = data_obj.timeseries_masked_dict[key].shape[0]

        if n_tp_actual == n_tp_expected:
            # ── scan is uncensored: apply censoring first to remove motion spikes
            print(f"{key}: full duration detected, applying censoring first")
            scan_censored   = data_obj.timeseries_masked_dict[key][censor_mask.astype(bool)]
            scan_uncensored = rubella.basic.uncensor_funcScan(scan_censored, censor_mask)
        else:
            # ── scan already censored: just uncensor to restore NaNs
            print(f"{key}: shorter duration detected, skipping censoring step")
            scan_uncensored = rubella.basic.uncensor_funcScan(
                data_obj.timeseries_masked_dict[key], censor_mask
            )

        # ── compute voxelwise baseline mean ───────────────────────────────────
        baseline_window = scan_uncensored[baseline_start:baseline_end, :]
        baseline_mean   = np.nanmean(baseline_window, axis=0)             # (n_voxels,)

        # ── express as percent change relative to baseline ────────────────────
        epsilon             = 1e-8
        scan_uncensored_pct = (scan_uncensored / (baseline_mean[np.newaxis, :] + epsilon) - 1) * 100

        # ── reapply censoring and save as nifti ───────────────────────────────
        scan_pct    = scan_uncensored_pct[censor_mask.astype(bool)]

        n_valid_tp  = scan_pct.shape[0]
        scan_pct_3d = np.zeros((n_valid_tp, *mask_arr_sitk.shape))
        scan_pct_3d[:, mask_arr_sitk] = scan_pct

        out_frames = []
        for t in range(n_valid_tp):
            frame = sitk.GetImageFromArray(scan_pct_3d[t].astype(np.float32))
            frame.CopyInformation(ref_sitk)
            out_frames.append(frame)

        out_sitk = sitk.JoinSeries(out_frames)
        out_path = os.path.join(output_dir, f"{key}_scaled-psc.nii.gz")
        sitk.WriteImage(out_sitk, out_path)
        print(f"Saved: {out_path}")

def dict_to_longform_fast(data_dict, total_time=60,  inj_time=30, roi_bool=False, roi_label_df = None, signal = "CBV"):
    ''' Convert from a dict of subjects (like from the data_obj) to a long df containing the signal '''
    # Extract subject list and ROI list
    subjects = list(data_dict.keys())

    if roi_bool == False:
        ts_list = list(data_dict.values())
        arr = np.stack(ts_list)   # shape: (S, T)
        print('no roi')
        print(arr.shape)
        S, T = arr.shape
        
        # Create shared time axis
        time_axis = np.linspace(0-inj_time, total_time-inj_time, T)

        # Vectorized assembly of the long-form DataFrame
        df = pd.DataFrame({
            "subject": np.repeat(subjects, T),
            "time": np.tile(time_axis, S),
            "CBV": arr.reshape(-1)
        })
    else:
        rois = list(next(iter(data_dict.values())).keys())

        # Build a 3D array: shape = (n_subjects, n_rois, n_timepoints)
        arr = np.stack([
            np.stack([data_dict[subj][roi] for roi in rois])
            for subj in subjects
        ])

        # Dimensions
        print('with roi')
        print(arr.shape)
        S, R, T = arr.shape

        # Create shared time axis
        time_axis = np.linspace(0-inj_time, total_time-inj_time, T)

        # Vectorized assembly of the long-form DataFrame
        df = pd.DataFrame({
            "subject": np.repeat(subjects, R * T),
            "roi_num": np.tile(np.repeat(rois, T), S),
            "time": np.tile(time_axis, S * R),
            signal: arr.reshape(-1),
        })
        # Optional ROI labels
        if roi_label_df is not None:
            roi_to_label = roi_label_df.set_index("parcel_ids")["acro"].to_dict()
            df["roi_label"] = df["roi_num"].map(roi_to_label)

    return df

def prepare_df_for_plotting_cbv(data_obj, roi_file, list_rois, roi_label_df, 
                                dataset_df, 
                                total_time, inj_time):
    '''Extract the global signal and signal within each ROI

    roi_file: singular nifti path to the commonspace roi labels
    list_rois: list of the roi numbers (integers) from the file that you want to plot
    total_time: total length of scan in seconds'''

    #load the censoring csvs
    censoring_mask_dict = data_obj.censoring_mask

    # load all the epis from my dataset into a dictionary
    timeseries_flat = data_obj.timeseries_masked_dict
    roi_file_flat = data_obj.load_img_to_array(roi_file) #flattened 2D array

    #uncensor to convert back to the original number of timepoints for point by point comparison - censored points replaced with nan
    uncensored_timeseries = {}
    uncensored_timeseries_mean = {}
    uncensored_timeseries_roi_mean = {}

    #loop over scans 
    for key in timeseries_flat.keys():
        #uncensor
        uncensored_timeseries[key] = rubella.basic.uncensor_funcScan(timeseries_flat[key],censoring_mask_dict[key])

        #compute the average timeseries over all voxels
        uncensored_timeseries_mean[key] = np.mean(uncensored_timeseries[key], axis = 1) #Take mean over all voxels

        #compute avg timeseries within roi
        uncensored_timeseries_roi_mean[key] = {} #for each sub, create a dict for all rois
        for roi_num in list_rois:
            mask_for_specific_roi = roi_file_flat == roi_num # find the indices for the specific roi of interest (bool array) from the roi label file
            timeseries_in_roi = uncensored_timeseries[key][:,mask_for_specific_roi] #extract the voxels within that roi
            uncensored_timeseries_roi_mean[key][roi_num] = np.mean(timeseries_in_roi, axis = 1) #Take mean over all voxels in roi

    #rearrange dict to long form
    df_long = dict_to_longform_fast(uncensored_timeseries_mean, total_time=total_time, inj_time = inj_time, roi_bool=False, roi_label_df = None) #shape of timeseries_mean is (ntimepoints,)
    df_long_roi = dict_to_longform_fast(uncensored_timeseries_roi_mean, total_time=total_time, inj_time = inj_time, roi_bool=True, roi_label_df=roi_label_df)#shape of timeseries_roi_mean is (ntimepoints,)

    #add the group info to the df
    dataset_df = dataset_df.rename(columns={"scan_id": "subject"})
    df_long = df_long.merge(dataset_df[['subject', 'group']], on='subject', how='left')
    df_long_roi = df_long_roi.merge(dataset_df[['subject', 'group']], on='subject', how='left')

    return uncensored_timeseries, df_long, df_long_roi 

def crop_and_bin_for_stats(data_obj, dataset_csv, nifti_savepath, path_to_template_mask,
                           TR, inj_time_min, starttime_min, endtime_min, window_duration_s, max_censored_per_window):
    timeseries_dict = data_obj.timeseries_dict
    censoring_mask = data_obj.censoring_mask
    starttimepoint = int(starttime_min*60.0/TR)
    endtimepoint = int(endtime_min*60.0/TR)
    dataset_df = pd.read_csv(dataset_csv, dtype={"sub": str})
    windowed_dataset_df_dict = {}

    for scan_id in timeseries_dict.keys():
        print(scan_id)

        #restore the original length of the scan before censoring by replacing the censored timepoints with nan placeholders
        print(f'shape original timeseries {timeseries_dict[scan_id].shape}')
        uncensored_scan = rubella.basic.uncensor_funcScan(timeseries_dict[scan_id], censoring_mask[scan_id])

        #crop the timeseries to a time interval of interest. also do same to mask.
        cropped_scan = uncensored_scan[starttimepoint:endtimepoint, ]
        cropped_censor_mask = censoring_mask[scan_id][starttimepoint:endtimepoint, ]

        #bin timeseries by averaging within window
        print(f'shape cropped scan {cropped_scan.shape}')
        window_avg_scan, valid_window_idx, invalid_window_idx, n_censored, window_starts, window_ends = rubella.basic.windowed_mean(cropped_scan, cropped_censor_mask, 
                                                                                                                        window_duration_s, TR, max_censored_per_window)
        print('num censored frames per window: ' + str(n_censored))
        print('discarded windows: ' + str(invalid_window_idx))

        #create a censoring mask on a window level. default is True (all windows kept), the replace the invalid ones with false
        censoring_mask_windows = np.ones(len(n_censored), dtype=bool)
        censoring_mask_windows[invalid_window_idx] = False
        cens_df = pd.DataFrame({'False = Masked Frames': censoring_mask_windows})
        cens_df.to_csv(nifti_savepath + f'/window_censoring_mask/{scan_id}_window_censoring_mask.csv')

        #convert window_starts and ends  to realtime after scan start (account for fact that it has been censored)
        window_starts_sinceScan = [x+(starttime_min*60) for x in window_starts]
        window_ends_sinceScan = [x+(starttime_min*60) for x in window_ends]

        #save each window avg as it's own nifti in the folder nifti_savepath. the name of the nifti should be scan_id and the window starttime that it corresponds to
        nifti_paths = []

        # save the binned timeseries in one nifti with n_window 'timepoints'
        print(f"window_avg_scan shape: {window_avg_scan.shape}")
        print(f"window_avg_scan min/max: {np.nanmin(window_avg_scan)}, {np.nanmax(window_avg_scan)}")
        img_series = nb.Nifti1Image(window_avg_scan.T, nb.load(path_to_template_mask).affine)
        nb.save(img_series, nifti_savepath + f'/binned_timeseries/{scan_id}_windows.nii.gz')

        print(f"IMG_SERIES shape: {img_series.shape}")
        print(f"img_series min/max: {np.nanmin(window_avg_scan.T)}, {np.nanmax(window_avg_scan.T)}")

        # save each individual window (vol) as a separate nifti
        for i, vol in enumerate(window_avg_scan):
            start_t = window_starts_sinceScan[i]

            img = nb.Nifti1Image(vol.T, nb.load(path_to_template_mask).affine, nb.load(path_to_template_mask).header)
            fpath = nifti_savepath +  f"/binned_timeseries_windows/{scan_id}_WindowAvgStartTime-{int(start_t)}s.nii.gz"
            nb.save(img, fpath)

            nifti_paths.append(fpath)
        # also extract the row of the dataset_df that matches scan_id (there's a column scan_id in the csv). now create a new df for each window_avg nifti.
        # one column shoud contain the path to each window_avg_nifti saved in the previous step. another column should contain window_starts and another window_ends
        # create also a binary column that is 0 for all window_starts that are before inj_time and 1 for all window_starts after it
        # the remaining columns are copied identically from dataset_df
        #save this dataframe in a dict
        #windowed_dataset_df_dict[scan_id] = windowed_dataset_df
        scan_row = dataset_df[dataset_df["scan_id"] == scan_id]

        if scan_row.shape[0] != 1:
            raise ValueError(f"scan_id {scan_id} not uniquely found in dataset_csv")

        scan_row = scan_row.iloc[0]

        rows = []
        for i in range(len(nifti_paths)):
            row_dict = scan_row.to_dict()

            row_dict["nifti_path"] = nifti_paths[i]
            row_dict["window_start"] = window_starts_sinceScan[i]
            row_dict["window_end"] = window_ends_sinceScan[i]
            row_dict["n_censored_frames"] = n_censored[valid_window_idx][i]
            
            # Binary pre/post injection
            row_dict["post_injection"] = int(window_starts_sinceScan[i] >= inj_time_min*60)

            rows.append(row_dict)

        windowed_dataset_df = pd.DataFrame(rows)

        windowed_dataset_df_dict[scan_id] = windowed_dataset_df
    
    #outside the for loop merge all of the windowed_dataset_df from windowed_dataset_df_dict across keys, into one master df and save it
    master_df = pd.concat(windowed_dataset_df_dict.values(), ignore_index=True)
    out_csv = os.path.join(nifti_savepath, "fusi_df_windowed_niftis.csv")
    master_df.to_csv(out_csv, index=False)

def plot_cbv_by_group(df_long, inj_time, yaxis_label = 'CBV', plot_sd = False, fig_aspect = 1.5):
    '''Plot the CBV or BOLD timeseries for each group (overlay all subjects one on top of the other)'''
    
    sns.set_context("poster")
    ################ get a separate color palette for each group so that colors are not reused across subjects from diff groups #####################
    group_subjects = df_long.groupby("group")["subject"].unique()

    # Cycle through specific color palettes for each group
    base_colors = [
        np.array([240, 128, 128]) / 255,  # salmon
        np.array([220, 130,  60]) / 255   # orange
    ]

    def make_shades(base_color, n):
        # Create shades by interpolating toward white
        factors = np.linspace(0.3, 1.0, n) 
        shades = [base_color * f + (1 - f) * np.array([1, 1, 1]) for f in factors]
        return shades

    group_palettes = {}

    for i, (group, subjects) in enumerate(group_subjects.items()):
        base = base_colors[i % len(base_colors)]
        palette = make_shades(base, len(subjects))
        group_palettes[group] = dict(zip(subjects, palette))

    # Merge into one big palette mapping every subject → unique color
    full_palette = {subj: color for g in group_palettes.values() for subj, color in g.items()}

    ##################### define plot ########################################
    g = sns.FacetGrid(df_long, col='group', sharey=True, height=5, aspect=fig_aspect)

    def plot_group(data, **kwargs):
        # Individual subject lines
        sns.lineplot(
            data=data, x='time', y='CBV', hue='subject', palette=full_palette, 
            legend=True, alpha=0.6, **kwargs
        )
        # Mean ± SD
        mean_series = data.groupby('time')['CBV'].mean()
        time_axis = mean_series.index
        plt.plot(time_axis, mean_series, color='black', linewidth=2)
        if plot_sd:
            sd_series = data.groupby('time')['CBV'].std()
            plt.fill_between(time_axis, mean_series - sd_series, mean_series + sd_series,
                            color='gray', alpha=0.3)
        # Dotted vertical line at center for the injection
        plt.axvline(inj_time, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(5, color='black', linestyle='--', linewidth=2.5)
        plt.axvline(10, color='gray', linestyle='--', linewidth=2.5)
        plt.axvline(15, color='gray', linestyle='--', linewidth=2.5)
        plt.axvline(20, color='gray', linestyle='--', linewidth=2.5)
        plt.axvline(25, color='gray', linestyle='--', linewidth=2.5)
        plt.axvline(30, color='gray', linestyle='--', linewidth=2.5)

    g.map_dataframe(plot_group)

    ################ manually add legend based on palette info ###########
    #usual plt.legend only plots the last facet
    handles = [mpatches.Patch(color=color, label=subj) for subj, color in full_palette.items()]
    plt.legend(
        handles=handles,
        title="Subject",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    
    g.set_axis_labels("Time (min)", yaxis_label + " (%)")
    g.set_titles("{col_name}")
    plt.show()

    return g

def plot_cbv_by_group_and_roi(df_long, inj_time, yaxis_label = 'CBV', plot_sd=False, saline_trend_df = None,
                               ylims = [-10,2], plot_mean = True, fig_aspect = 2, time_window_stats_lines = False):
    """
    df_long must contain: time, subject, group, roi, CBV
    """
    import matplotlib.patches as mpatches
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_context("poster")
    print(sns.plotting_context("poster"))

    ###########################################################################
    # Assign unique colors per subject, consistent across all ROIs and groups
    ###########################################################################
    group_subjects = df_long.groupby("group")["subject"].unique()
    
    # Cycle through specific color palettes for each group
    base_colors = [
        np.array([240, 128, 128]) / 255,  # salmon
        np.array([220, 130,  60]) / 255   # orange
    ]

    def make_shades(base_color, n):
        # Create shades by interpolating toward white
        factors = np.linspace(0.3, 1.0, n) 
        shades = [base_color * f + (1 - f) * np.array([1, 1, 1]) for f in factors]
        return shades

    group_palettes = {}

    for i, (group, subjects) in enumerate(group_subjects.items()):
        base = base_colors[i % len(base_colors)]
        palette = make_shades(base, len(subjects))
        group_palettes[group] = dict(zip(subjects, palette))

    # Merge into one big palette mapping every subject → unique color
    full_palette = {subj: color for g in group_palettes.values() for subj, color in g.items()}
    ###########################################################################
    # Create FacetGrid: ROIs in rows, groups in columns
    ###########################################################################
    g = sns.FacetGrid(
        df_long,
        row="roi_label",
        col="group",
        sharey=False,
        sharex= False,
        height=4,
        aspect=fig_aspect,
        margin_titles=True
    )

    ###########################################################################
    # Plotting function for each facet
    ###########################################################################
    def plot_group(data, **kwargs):
            
            # Mean and SD
            mean_series = data.groupby('time')['CBV'].mean()
            sd_series   = data.groupby('time')['CBV'].std()
            time_axis   = mean_series.index

            if plot_sd == False:
                # Plot individual subject traces
                sns.lineplot(
                    data=data,
                    x='time',
                    y='CBV',
                    hue='subject',
                    palette=full_palette,
                    legend=False,
                    alpha=0.5,
                    **kwargs
                )
            else :
                plt.fill_between(
                    time_axis,
                    mean_series - sd_series,
                    mean_series + sd_series,
                    color='gray',
                    alpha=0.3
                )
            #plot the mean on top
            if plot_mean:
                plt.plot(time_axis, mean_series, color='black', linewidth=1)
            
            # overlay the saline trend for the current ROI
            if saline_trend_df is not None:
                current_roi = data['roi_label'].iloc[0]
                trend_subset = saline_trend_df[saline_trend_df['roi_label'] == current_roi]
                if not trend_subset.empty:
                    plt.plot(trend_subset['time'], trend_subset['CBV'],
                            color='darkred', linewidth=3.5, linestyle='--', label='saline trend')

            plt.axvline(inj_time, color='gray', linestyle='--', linewidth=2.5)
            if time_window_stats_lines:
                #indicate in which windows the stats were done
                plt.axvline(inj_time, color='black', linestyle='--', linewidth=2.5)
                plt.axvline(5, color='gray', linestyle='--', linewidth=2.5)
                plt.axvline(10, color='gray', linestyle='--', linewidth=2.5)
                plt.axvline(15, color='gray', linestyle='--', linewidth=2.5)
                plt.axvline(20, color='gray', linestyle='--', linewidth=2.5)
                plt.axvline(25, color='gray', linestyle='--', linewidth=2.5)
                plt.axvline(30, color='gray', linestyle='--', linewidth=2.5)
            plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
            plt.ylim(ylims)
    g.map_dataframe(plot_group)

    ###########################################################################
    # Manual legend (global)
    ###########################################################################
    handles = [
        mpatches.Patch(color=color, label=subj)
        for subj, color in full_palette.items()
    ]

    g.fig.legend(
        handles=handles,
        title="Subject",
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),  # move far right so it never overlaps
        frameon=False,
        borderaxespad=0.,
        fontsize='small',            # smaller labels
        title_fontsize='medium'
    )
    ###########################################################################
    # Axis labels, titles
    ###########################################################################
    g.set_axis_labels("Time (min)", yaxis_label + " (%)")
    g.set_titles(row_template="ROI {row_name}", col_template="{col_name}")

    plt.tight_layout()
    plt.close()
    return g

def plot_cbv_by_group_and_roi_groupmean(df_long, inj_time, yaxis_label='CBV', plot_sd=False):
    import numpy as np
    import pandas as pd
    import matplotlib.patches as mpatches
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_context("poster")

    ###########################################################################
    # --- CREATE DIFFERENCE DATA (saline - exp) ---
    ###########################################################################
    mean_df = (
        df_long
        .groupby(['roi_label', 'group', 'time'])['CBV']
        .mean()
        .reset_index()
    )

    # Pivot to compute difference
    pivot_df = mean_df.pivot_table(
        index=['roi_label', 'time'],
        columns='group',
        values='CBV'
    ).reset_index()

    # Adjust names if needed
    if 'saline' in pivot_df.columns and 'exp' in pivot_df.columns:
        pivot_df['CBV'] =  pivot_df['exp'] - pivot_df['saline']
    else:
        raise ValueError("Expected groups 'saline' and 'exp'")

    pivot_df['group'] = 'diff'
    pivot_df['subject'] = 'diff'  # dummy
    pivot_df = pivot_df[['roi_label', 'time', 'CBV', 'group', 'subject']]

    # Combine with original data
    df_plot = pd.concat([df_long, pivot_df], ignore_index=True)

    ###########################################################################
    # Assign unique colors per subject (exclude diff)
    ###########################################################################
    group_subjects = df_plot[df_plot['group'] != 'diff'].groupby("group")["subject"].unique()

    base_colors = [
        np.array([240, 128, 128]) / 255,
        np.array([220, 130, 60]) / 255
    ]

    def make_shades(base_color, n):
        factors = np.linspace(0.3, 1.0, n)
        return [base_color * f + (1 - f) * np.array([1, 1, 1]) for f in factors]

    group_palettes = {}
    for i, (group, subjects) in enumerate(group_subjects.items()):
        base = base_colors[i % len(base_colors)]
        palette = make_shades(base, len(subjects))
        group_palettes[group] = dict(zip(subjects, palette))

    full_palette = {subj: color for g in group_palettes.values() for subj, color in g.items()}

    ###########################################################################
    # FacetGrid (now includes diff column)
    ###########################################################################
    g = sns.FacetGrid(
        df_plot,
        row="roi_label",
        col="group",
        sharey=True,
        height=3,
        aspect=1.5,
        margin_titles=True,
        col_order=['saline', 'exp', 'exp-saline']  # force order
    )

    ###########################################################################
    # Plot function
    ###########################################################################
    def plot_group(data, **kwargs):

        group_name = data['group'].iloc[0]

        if group_name != 'diff':
            # Individual subjects
            sns.lineplot(
                data=data,
                x='time',
                y='CBV',
                hue='subject',
                palette=full_palette,
                legend=False,
                alpha=0.6,
                **kwargs
            )

            mean_series = data.groupby('time')['CBV'].mean()
            sd_series   = data.groupby('time')['CBV'].std()

            plt.plot(mean_series.index, mean_series, color='black', linewidth=2)

            if plot_sd:
                plt.fill_between(
                    mean_series.index,
                    mean_series - sd_series,
                    mean_series + sd_series,
                    color='gray',
                    alpha=0.3
                )

        else:
            # Difference plot → single line
            mean_series = data.groupby('time')['CBV'].mean()
            plt.plot(mean_series.index, mean_series, color='blue', linewidth=2)
            plt.axhline(0, color='black', linestyle='--', linewidth=2.5)

        # Injection line
        plt.axvline(inj_time, color='black', linestyle='--', linewidth=2.5)

    g.map_dataframe(plot_group)

    ###########################################################################
    # Legend (exclude diff)
    ###########################################################################
    handles = [
        mpatches.Patch(color=color, label=subj)
        for subj, color in full_palette.items()
    ]

    g.fig.legend(
        handles=handles,
        title="Subject",
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        frameon=False,
        fontsize='small',
        title_fontsize='medium'
    )

    ###########################################################################
    # Labels
    ###########################################################################
    g.set_axis_labels("Time (min)", yaxis_label + " (%)")
    g.set_titles(row_template="ROI {row_name}", col_template="{col_name}")

    plt.tight_layout()
    plt.show()

    return g

def subject_level_timeseries_QC(list_dfs, list_df_detrending_type, list_df_timewindow_type, inj_time, fig_save_path,
                                ylims = [-10,4], yaxis_label='CBV', df_trend_global=None, df_trend_rois=None, nytick=3, nxtick=4):
    sns.set_style("ticks")   # call this first
    sns.set_context("poster")
    plt.rcParams.update({'axes.linewidth': 3.5, 
                            'grid.linewidth': 2, 
                            'lines.linewidth': 4.0, 
                            'lines.markersize': 14, 
                            'patch.linewidth': 2, 
                            'xtick.major.width': 2.5, 
                            'ytick.major.width': 2.5, 
                            'xtick.minor.width': 2, 
                            'ytick.minor.width': 2, 
                            'xtick.major.size': 12, 
                            'ytick.major.size': 12, 
                            'xtick.minor.size': 8, 
                            'ytick.minor.size': 8, 
                            'font.size': 26, 
                            'axes.labelsize': 26, 
                            'axes.titlesize': 26, 
                            'xtick.labelsize': 24, 
                            'ytick.labelsize': 24
                            })

    plot_color = {'exp': np.array([240, 128, 128]) / 255,
                  'saline': np.array([220, 130, 60]) / 255}

    df_long = pd.concat([
        df.assign(**{'detrending type': det_name, 'time window type': tw_name})
        for df, det_name, tw_name in zip(list_dfs, list_df_detrending_type, list_df_timewindow_type)
    ], ignore_index=True)

    col_order = list(dict.fromkeys(list_df_detrending_type))  # unique, order-preserving
    row_order = [r for r in df_long['roi_label'].unique() if pd.notna(r)]
    n_cols    = len(col_order)
    n_rows    = len(row_order)

    for sub in df_long['subject'].unique():
        print(sub)

        sub_specific_df = df_long.loc[df_long['subject'] == sub]
        group           = sub_specific_df['group'].unique()[0]
        color           = plot_color[group]

        df_timepoint = sub_specific_df[sub_specific_df['time window type'] == 'timepoint']
        df_1min      = sub_specific_df[sub_specific_df['time window type'] == '1-min window']

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(8 * n_cols, n_rows * (4 + 0.2)),
                                 squeeze=False)
        
        for ax in axes.ravel():
            sns.despine(ax=ax) #remove right and upper border to match seaborn formatting in other plots
            ax.xaxis.set_major_locator(plt.MaxNLocator(nxtick)) # match number of x-ticks to seaborn formatting
            ax.yaxis.set_major_locator(plt.MaxNLocator(nytick))

        for i, roi in enumerate(row_order):
            # match seaborn's margin_titles style — row labels on the right
            axes[i, -1].annotate(roi, xy=(1.02, 0.5), xycoords='axes fraction',
                                 fontsize=plt.rcParams['font.size'],
                                 va='center', rotation=270)
            
            for j, det in enumerate(col_order):
                axes[0, j].set_title(det, fontsize=plt.rcParams['font.size']) # col titles on top
                ax   = axes[i, j]
                ax.set_ylim(ylims)

                # main timeseries
                subset_tp = df_timepoint[
                    (df_timepoint['roi_label'] == roi) &
                    (df_timepoint['detrending type'] == det)
                ]
                if not subset_tp.empty:
                    ax.plot(subset_tp['time'], subset_tp['CBV'],
                            color=color, linewidth=1)

                # 1-min window overlay
                subset_1min = df_1min[
                    (df_1min['roi_label'] == roi) &
                    (df_1min['detrending type'] == det)
                ]
                if not subset_1min.empty:
                    ax.step(subset_1min['time'], subset_1min['CBV'], where='post',
                            color='black', linewidth=1.5, label='1-min window')

                # trend overlay
                if det == 'no detrend':
                    if roi == 'global' and df_trend_global is not None:
                        ax.plot(df_trend_global['time'], df_trend_global['CBV'],
                                color='darkred', linewidth=2.5, linestyle='--', label='saline trend')
                    elif roi != 'global' and df_trend_rois is not None:
                        trend_subset = df_trend_rois[df_trend_rois['roi_label'] == roi]
                        if not trend_subset.empty:
                            ax.plot(trend_subset['time'], trend_subset['CBV'],
                                    color='darkred', linewidth=2.5, linestyle='--', label='saline trend')
                else:
                    ax.axhline(0, color='black', linewidth=1.5, linestyle='--')

                ax.axvline(inj_time, color='gray', linestyle='--', linewidth=2.5)
                ax.set_ylabel(yaxis_label + " (%)")

                # row and col labels only on edges
                if j == 0:
                    ax.set_ylabel(f'{yaxis_label} (%)', fontsize=plt.rcParams['font.size'])
                else:
                    ax.set_ylabel('')
                    
                if i == n_rows - 1:
                    ax.set_xlabel('Time (min)', fontsize=plt.rcParams['font.size'])
                else:
                    ax.set_xlabel('')

        fig.suptitle(sub, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{fig_save_path}/{sub}_timeseries_QC.png',
                    bbox_inches='tight', pad_inches=0.3, transparent=False)
        plt.close()