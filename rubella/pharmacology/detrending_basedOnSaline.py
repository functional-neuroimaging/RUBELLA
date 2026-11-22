import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import SimpleITK as sitk
from nilearn.plotting import plot_stat_map 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys
sys.path.append('/media/DATA2/fUS/mila/code/RUBELLA')
import rubella.basic
import rubella.pharmacology.pharmacology_main

def compute_trend_in_saline(data_obj, saline_keys, save_path: str):

    """
    Computes the trend across (3rd degree poly) in saline scans, voxelwise.
    Outputs a dictionary, including voxelwise coefficients for each of the 4 terms in the polynomial.
    
    Parameters
    ----------
    data_obj : the obj loaded by rubella.data_obj

    saline_keys : list or str
        Either:
        - list of scan IDs
        - path to csv containing columns:
            'Group' (saline or other), exclude_during_QC (bool) and 'scan_id' (i.e. a csv containing the dataset info)

    """

    # ── Convert csv -> saline_keys list if needed ──────────────────────────
    if isinstance(saline_keys, str):

        df = pd.read_csv(saline_keys)

        saline_keys = (
            df.loc[(df['group'] == 'saline') & (df['exclude_during_QC'] == False), 'scan_id']
            .tolist()
        )

        print(f'list of included saline scans: {saline_keys}')

    #load the censoring csvs - each key is a subject and each value a 1d array of shape time
    censoring_mask_dict = data_obj.censoring_mask

    # ── 1. Uncensor scans to restore original number of timepoints ───────────────────
    # load all the epis from my dataset into a dictionary - each key is a subject and each value a 2d array of time x voxels
    timeseries_flat = data_obj.timeseries_masked_dict

    #uncensor to convert back to the original number of timepoints for point by point comparison - censored points replaced with nan
    uncensored_timeseries = {}

    #loop over scans based on the list of censoring csvs
    for key in saline_keys:
        #uncensor
        uncensored_timeseries[key] = rubella.basic.uncensor_funcScan(timeseries_flat[key],censoring_mask_dict[key])

    # ── 2. STACK ALL SCANS → (total_valid_timepoints, n_voxels) ───────────────────
    # Each scan contributes its non-NaN timepoints as independent observations
    # at their original time indices. This lets the polynomial fit the temporal
    # trend across time, using all scans as replicated evidence.

    scan_keys   = list(uncensored_timeseries.keys())
    n_timepoints = uncensored_timeseries[scan_keys[0]].shape[0]   # full uncensored length (e.g. 3600)
    n_voxels     = uncensored_timeseries[scan_keys[0]].shape[1]

    t_full  = np.arange(n_timepoints, dtype=float)
    t_norm  = (t_full - t_full.mean()) / (t_full.std() + 1e-8)

    # Collect valid rows from every scan
    all_t_idx  = []   # which timepoint index each row came from
    all_Y      = []   # corresponding signal, shape (n_valid_rows, n_voxels)

    for key in scan_keys:
        data = uncensored_timeseries[key]            # (3600, n_voxels), NaNs at censored points
        # a timepoint is valid if it has no NaN across voxels
        valid_mask = ~np.isnan(data).any(axis=1)    # (3600,)
        all_t_idx.append(np.where(valid_mask)[0])
        all_Y.append(data[valid_mask])

    all_t_idx = np.concatenate(all_t_idx)           # (total_valid_rows,)
    all_Y     = np.vstack(all_Y)                    # (total_valid_rows, n_voxels)

    # ── 3. FIT 3RD-ORDER POLYNOMIAL, VOXELWISE ────────────────────────────────────
    t_for_fit = t_norm[all_t_idx]                   # normalised x for each valid row
    V = np.vander(t_for_fit, N=4, increasing=True)  # (total_valid_rows, 4)

    # single lstsq solves all voxels at once: coeffs shape (4, n_voxels)
    coeffs, _, _, _ = np.linalg.lstsq(V, all_Y, rcond=None)

    # ── 5. RETURN everything needed to apply the trend to any scan ─────────────
    # coefficients  : (4, n_voxels) — the fitted β maps; the core output
    # t_norm        : (3600,)       — normalised time axis; needed to reconstruct
    #                                 the trend at any timepoint via V @ coeffs
    # n_timepoints  : int           — length of the full uncensored time axis
    result = {
        'coefficients' : coeffs,
        't_norm'       : t_norm,
        'n_timepoints' : n_timepoints,
    }

    full_save_path = save_path + '/fitted_trend_saline.pkl'
    with open(full_save_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved detrend info to {full_save_path}")

    return result

def save_voxelwise_trend_as_nifti(detrend_info: dict, data_obj, reference_nifti: str, output_path: str,
                                  roi_file, list_rois, roi_label_df, total_time, inj_time):
    """
    Reconstruct the fitted voxelwise polynomial trend across time and save as a 4D nifti.
    
    Parameters
    ----------
    detrend_info     : dict output from compute_trend_in_saline()
    data_obj         : must have .volume_idx — 3D bool sitk mask 
    reference_nifti  : path to a 3D nifti for spatial metadata (spacing, origin, direction)
    output_path      : folder path for the output nifti 
    """
    t_norm   = detrend_info['t_norm']           # (n_timepoints,)
    n_tp     = detrend_info['n_timepoints']
    coeffs   = detrend_info['coefficients']     # (4, n_voxels)
    mask     = data_obj.volume_idx              # (x,y,z), bool
    roi_file_flat = data_obj.load_img_to_array(roi_file) #flattened 2D array

    # reconstruct the full temporal trend for every voxel
    V     = np.vander(t_norm[:n_tp], N=4, increasing=True)  # (n_tp, 4)
    trend = V @ coeffs                                        # (n_tp, n_voxels)

    # unflatten to (n_tp, x,y,z)
    trend_3d = np.zeros((n_tp, *mask.shape), dtype=np.float32)
    trend_3d[:, mask] = trend

    # save using sitk with metadata from reference
    ref_sitk  = sitk.ReadImage(reference_nifti)
    out_frames = []
    for t in range(n_tp):
        frame = sitk.GetImageFromArray(trend_3d[t])
        frame.CopyInformation(ref_sitk)
        out_frames.append(frame)

    out_sitk = sitk.JoinSeries(out_frames)
    sitk.WriteImage(out_sitk, output_path + '/trend_in_saline.nii.gz')
    print(f"Saved voxelwise trend to {output_path + '/trend_in_saline.nii.gz'}")

    # also compute the global signal and roi signals and save in a csv for later plotting
    # this was already part of the prepare_df_for_plotting_cbv function, but more efficient to also include here to avoid realoding data_obj
    trend_globalAvg = np.mean(trend, axis = 1) #Take mean over all voxels

    #compute avg timeseries within roi (pre and post detrending)
    trend_roiAvg = {} #for each sub, create a dict for all rois
    for roi_num in list_rois:
        mask_for_specific_roi = roi_file_flat == roi_num # find the indices for the specific roi of interest (bool array) from the roi label file
        
        timeseries_in_roi = trend[:,mask_for_specific_roi] #extract the voxels within that roi
        trend_roiAvg[roi_num] = np.mean(timeseries_in_roi, axis = 1) #Take mean over all voxels in roi

    ################### save trend info in dataframe ###########################
    #similar to dict_to_longform_fast function, except doesn't iterate over keys in a dict
    # global trend df
    time_axis = np.linspace(0 - inj_time, total_time - inj_time, n_tp)
    
    df_long_trend = pd.DataFrame({'group':'saline','time': time_axis, 'CBV': trend_globalAvg })

    # roi trend df
    roi_rows = []
    for roi_num, roi_ts in trend_roiAvg.items():
        roi_df = pd.DataFrame({'group' : 'saline','time' : time_axis,'CBV' : roi_ts,'roi_num': roi_num })
        roi_rows.append(roi_df)
    df_long_roi_trend = pd.concat(roi_rows, ignore_index=True)

    # optional roi labels
    if roi_label_df is not None:
        roi_to_label = roi_label_df.set_index('parcel_ids')['acro'].to_dict()
        df_long_roi_trend['roi_label'] = df_long_roi_trend['roi_num'].map(roi_to_label)

    return df_long_trend, df_long_roi_trend

def _plot_detrend_maps(detrending_fit, data_obj, template_file, savepath, coords = (1,2,3,4,5)):
    """
    Plot the spatial map of each polynomial coefficient from the detrending as a coronal slice
    using nilearn's plot_stat_map. This allows to see how the shape of trend varies across voxels.

    detrending_fit   : dict output from detrend_based_on_saline function )
    data_obj : must have:
                 .brain_mask_file  — path to the brain mask
                 .volume_idx       — 3D nibabel image, True/1 at brain voxels
    """
    coeffs = detrending_fit['coefficients']
    labels = ['β₀  intercept', 'β₁  linear', 'β₂  quadratic', 'β₃  cubic']

    mask_img     = nb.load(data_obj.brain_mask_file)
    template_img = nb.load(template_file)

    # use the sitk brain mask directly for unflattening — already (41, 42, 53) and boolean
    mask_arr_sitk = data_obj.volume_idx              # (41, 42, 53), dtype bool


    fig, axes = plt.subplots(4, 1, figsize=(7, 7))

    for i, (ax, label) in enumerate(zip(axes, labels)):

        # unflatten into sitk space using the sitk mask
        coef_vol_sitk = np.zeros(mask_arr_sitk.shape)
        coef_vol_sitk[mask_arr_sitk] = coeffs[i]

        # transpose to nibabel ordering (x, y, z) → (z, y, x)
        coef_vol_nib = coef_vol_sitk.transpose(2, 1, 0)        # (53, 42, 41)
        coef_img     = nb.Nifti1Image(coef_vol_nib, affine=mask_img.affine, header=mask_img.header)


        vmax = np.nanpercentile(np.abs(coeffs[i]), 90)

        plot_stat_map(coef_img,bg_img=template_img, display_mode='y',cut_coords = coords,colorbar=True,vmax=vmax,symmetric_cbar=True,title=label,axes=ax)

    fig.suptitle('Voxelwise polynomial detrend — coefficient spatial maps',fontsize=13, fontweight='bold')
    plt.savefig(f'{savepath}/detrending_coefficients_as_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    
def apply_detrend(data_obj, detrend_info: dict, output_dir: str, reference_nifti: str, 
                  dataset_df, roi_file, list_rois, roi_label_df, total_time, inj_time, TR):
    '''
    Applies voxelwise detrending to all scans in data_obj and saves the results as niftis.
    
    The processing steps per scan are:
        1. Uncensor the scan to restore the full time axis (censored points = NaN)
        2. Subtract the fitted 3rd-order polynomial trend (computed from saline scans)
        3. Reapply censoring and save as a 4D nifti

    Also saves global and ROI-averaged timeseries (pre and post detrending) as long-form
    dataframes for later plotting.

    Parameters
    ----------
    data_obj        : DataObject with .timeseries_masked_dict, .censoring_mask, .volume_idx
    detrend_info    : dict output from detrend_based_on_saline()
    output_dir      : directory where detrended niftis will be saved
    reference_nifti : path to a 3D nifti for sitk spatial metadata
    dataset_df      : dataframe with columns scan_id and group, for merging group info
    roi_file        : path to a 3D nifti with roi labels
    list_rois       : list of roi label numbers to extract
    roi_label_df    : dataframe mapping roi numbers to roi names
    total_time      : total scan duration in minutes, for the time axis
    inj_time        : injection time in minutes, for the time axis
    TR              : repetition time in seconds
    '''

    os.makedirs(output_dir, exist_ok=True)

    mask_arr_sitk = data_obj.volume_idx
    t_norm        = detrend_info['t_norm']
    n_tp          = detrend_info['n_timepoints']

    ref_sitk = sitk.ReadImage(reference_nifti)

    # initialise all dicts
    uncensored_timeseries_mean_predetrend          = {}
    uncensored_timeseries_roi_mean_predetrend      = {}
    uncensored_timeseries_mean_postdetrend         = {}
    uncensored_timeseries_roi_mean_postdetrend     = {}
    roi_file_flat = data_obj.load_img_to_array(roi_file)

    for key in data_obj.censoring_mask.keys():

        # ── 1. uncensor ───────────────────────────────────────────────────────
        scan_uncensored = rubella.basic.uncensor_funcScan(
            data_obj.timeseries_masked_dict[key],
            data_obj.censoring_mask[key]
        )                                                      # (n_tp, n_voxels)

        # ── 2. subtract polynomial trend ─────────────────────────────────────
        V         = np.vander(t_norm[:n_tp], N=4, increasing=True)
        trend     = V @ detrend_info['coefficients']
        detrended_sub = scan_uncensored - trend                    # (n_tp, n_voxels)

        # ── global and roi averages (pre and post detrending, both versions) ───
        uncensored_timeseries_mean_predetrend[key]         = np.mean(scan_uncensored, axis=1)
        uncensored_timeseries_mean_postdetrend[key]        = np.mean(detrended_sub, axis=1)

        uncensored_timeseries_roi_mean_predetrend[key]      = {}
        uncensored_timeseries_roi_mean_postdetrend[key]     = {}

        for roi_num in list_rois:
            mask_for_specific_roi = roi_file_flat == roi_num
            uncensored_timeseries_roi_mean_predetrend[key][roi_num]      = np.mean(scan_uncensored[:, mask_for_specific_roi], axis=1)
            uncensored_timeseries_roi_mean_postdetrend[key][roi_num]     = np.mean(detrended_sub[:, mask_for_specific_roi], axis=1)

        # ── 5. reapply censoring and save both niftis ─────────────────────────
        censor_mask = data_obj.censoring_mask[key]


        detrended_censored = detrended_sub[censor_mask.astype(bool)]

        n_valid_tp   = detrended_censored.shape[0]
        detrended_3d = np.zeros((n_valid_tp, *mask_arr_sitk.shape))
        detrended_3d[:, mask_arr_sitk] = detrended_censored

        out_frames = []
        for t in range(n_valid_tp):
            frame = sitk.GetImageFromArray(detrended_3d[t].astype(np.float32))
            frame.CopyInformation(ref_sitk)
            out_frames.append(frame)

        out_sitk = sitk.JoinSeries(out_frames)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{key}_detrendedBasedOnSaline.nii.gz")
        sitk.WriteImage(out_sitk, out_path)
        print(f"Saved: {out_path}")

    # ── convert to long form ──────────────────────────────────────────────────
    df_long_predetrend          = rubella.pharmacology.pharmacology_main.dict_to_longform_fast(uncensored_timeseries_mean_predetrend, total_time=total_time, inj_time=inj_time, roi_bool=False, roi_label_df=None)
    df_long_roi_predetrend      = rubella.pharmacology.pharmacology_main.dict_to_longform_fast(uncensored_timeseries_roi_mean_predetrend, total_time=total_time, inj_time=inj_time, roi_bool=True, roi_label_df=roi_label_df)
    df_long_postdetrend         = rubella.pharmacology.pharmacology_main.dict_to_longform_fast(uncensored_timeseries_mean_postdetrend, total_time=total_time, inj_time=inj_time, roi_bool=False, roi_label_df=None)
    df_long_roi_postdetrend     = rubella.pharmacology.pharmacology_main.dict_to_longform_fast(uncensored_timeseries_roi_mean_postdetrend, total_time=total_time, inj_time=inj_time, roi_bool=True, roi_label_df=roi_label_df)

    # ── merge group info ──────────────────────────────────────────────────────
    dataset_df = dataset_df.rename(columns={"scan_id": "subject"})
    all_dfs = [df_long_predetrend, df_long_roi_predetrend,
            df_long_postdetrend, df_long_roi_postdetrend]
    for df in all_dfs:
        df[['group']] = df[['subject']].merge(dataset_df[['subject', 'group']], on='subject', how='left')[['group']]

    return (df_long_predetrend, df_long_roi_predetrend,
            df_long_postdetrend, df_long_roi_postdetrend)