import SimpleITK as sitk
import numpy as np
from rabies.analysis_pkg.analysis_math import dual_regression
import os
from pathlib import Path


def load_img_to_array(img_file, brain_mask_file, input_is_mask=False):
    """Load a Nifti image as an array, including only the voxels within the brain mask.

    The input brain mask determines the expected image array spatial dimensions and resolution.
    If the input ``img_file`` does not match the resolution of the brain mask, the image
    is resampled to match the brain mask volume before being loaded as an array.
    IMPORTANTLY: this assumes that despite a mismatch in resolution, the two images occupy the
    same space - if not, the loaded array is likely to be empty after resampling to the brain mask.

    Parameters
    ----------
    img_file : str or pathlib.Path
        Path to the input Nifti file.

    brain_mask_file : str or pathlib.Path
        Path to the brain mask Nifti file, which is used to determine brain voxel array indices.

    input_is_mask : bool, default=False
        Whether the input ``img_file`` is itself a brain mask or parcellation, in which case the
        resampling is carried using the antsApplyTransforms GenericLabel interpolator.

    Returns
    -------
    img_array : numpy.ndarray
        The output image array, 2D (time by voxel) for a 4D input image, or 1D for a 3D image.
    """
    img_file = Path(img_file)
    brain_mask_file = Path(brain_mask_file)
    sitk_img = sitk.ReadImage(img_file)
    dim = sitk_img.GetDimension()

    mask_img = sitk.ReadImage(brain_mask_file)
    volume_idx = sitk.GetArrayFromImage(mask_img).astype(bool)

    if not sitk_img.GetSize()[:3]==mask_img.GetSize():
        print("""
                Warning: The input image array does not match the brain mask.
                The image is resampled with the assumption that it occupies 
                the same space as the brain mask.
                """)
        if input_is_mask:
            from rabies.utils import antsApplyTransforms
            import tempfile, shutil
            tmppath = tempfile.mkdtemp()
            out_file = f"{tmppath}/resampled_mask.nii.gz"
            antsApplyTransforms(transforms = [], inverses = [], 
                input_image = str(img_file), ref_image = str(brain_mask_file), output_filename = out_file, interpolation='GenericLabel', rabies_data_type=sitk.sitkInt16, clip_negative=False)
            sitk_img = sitk.ReadImage(out_file, sitk.sitkInt16)
            shutil.rmtree(tmppath)
        else:
            from rabies.utils import resample_volumes
            sitk_img = resample_volumes(in_img=sitk_img, in_ref=mask_img, transforms_3d_files=[], inverses_3d=[], 
                                                    motcorr_params_file=None, interpolation=sitk.sitkLinear, rabies_data_type=sitk.sitkFloat32, clip_negative=False)

    if dim==4:
        img_array = sitk.GetArrayFromImage(sitk_img)[:,volume_idx]
    elif dim==3:
        img_array = sitk.GetArrayFromImage(sitk_img)[volume_idx]
    return img_array


def vcorrcoef(X, Y):  # return a correlation between each row of X with each row of Y
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = (X-Xm).dot((Y-Ym).T)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1).reshape(-1,1)*np.sum((Y-Ym)**2, axis=1).reshape(1,-1))
    r = r_num/r_den
    return r


def seed_based_FC(timeseries_dict, brain_mask_file, img_id_list, seed_roi, z_corr=False):
    """Compute seed-based connectivity for one input seed, for each image ID within a timeseries_dict.

    Parameters
    ----------
    timeseries_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated timeseries array,
        e.g. :attr:`rubella.data_obj.DataObject.timeseries_dict`.

    brain_mask_file : str or pathlib.Path
        The full path to the brain mask Nifti file,
        e.g. :attr:`rubella.data_obj.DataObject.brain_mask_file`.

    img_id_list : list of str
        List of image IDs that are keys for ``timeseries_dict``,
        e.g. :attr:`rubella.data_obj.DataObject.img_id_list`.

    seed_roi : str, pathlib.Path or numpy.ndarray
        The input seed region of interest. Must be a brain mask delineating the voxels part of
        the seed region. Can be either the full path to a Nifti file,
        or a pre-loaded 1D array of shape (n_voxels,).

    z_corr : bool, default=False
        Whether to output Fisher z transformed correlation values, instead of Pearson's r.

    Returns
    -------
    SBC_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated connectivity map.
    """

    brain_mask_file = Path(brain_mask_file)
    if isinstance(seed_roi, np.ndarray) or (not isinstance(seed_roi, np.ndarray) and Path(seed_roi).is_file()):
        seed_roi_arr = _parse_parcellation([seed_roi], brain_mask_file, roi_ids=None)[1]
    else:
        raise ValueError("seed_roi is neither an array nor a file.")

    SBC_dict = {}
    for img_id in img_id_list:
        timeseries = timeseries_dict[img_id]
        # extract the voxel timeseries within the mask, and take the mean ROI timeseries
        seed_timeseries = timeseries[:,seed_roi_arr].mean(axis=1)
        seed_FC = vcorrcoef(timeseries.T, seed_timeseries.reshape(1,-1)).flatten() # shape the output into a vector
        seed_FC[np.isnan(seed_FC)] = 0
        if z_corr:
            seed_FC = np.arctanh(seed_FC)
        SBC_dict[img_id] = seed_FC
    return SBC_dict
    

def compute_dual_regression(timeseries_dict, brain_mask_file, img_id_list, IC_maps, IC_idx=None, z_corr=False):
    """Compute dual regression outputs for each image ID within a timeseries_dict.

    Parameters
    ----------
    timeseries_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated timeseries array,
        e.g. :attr:`rubella.data_obj.DataObject.timeseries_dict`.

    brain_mask_file : str or pathlib.Path
        The full path to the brain mask Nifti file,
        e.g. :attr:`rubella.data_obj.DataObject.brain_mask_file`.

    img_id_list : list of str
        List of image IDs that are keys for ``timeseries_dict``,
        e.g. :attr:`rubella.data_obj.DataObject.img_id_list`.

    IC_maps : str, pathlib.Path or numpy.ndarray
        The input ICA maps, can be either the full path to a Nifti file,
        or a pre-loaded array of shape (n_ICs, n_voxels).

    IC_idx : list of int, default=None
        Optionally, a list of indices can be provided to pre-select the IC components
        of interest, and the outputs will only include those indices.
        If None, all components are included.

    z_corr : bool, default=False
        Whether to output Fisher z transformed correlation values for ``'DR_cross_corr'``.
        If False, the output is Pearson's r.

    Returns
    -------
    DR_dict : dict
        Nested dictionary with ``img_id_list`` as keys. Each entry is a dictionary with the
        following keys:

        ``'DR_maps'`` : numpy.ndarray of shape (n_voxels, n_ICs)
            The set of dual regression maps.
        ``'DR_time'`` : numpy.ndarray of shape (n_timepoints, n_ICs)
            The timecourses of the dual regression components.
        ``'DR_amp'`` : numpy.ndarray of shape (n_ICs,)
            The amplitude value estimated from each map.
        ``'DR_cross_corr'`` : numpy.ndarray of shape (n_ICs, n_ICs)
            A correlation matrix derived from the mutual correlation of
            all DR component timecourses. The diagonal is NaN.
    """

    brain_mask_file = Path(brain_mask_file)
    volume_idx = sitk.GetArrayFromImage(sitk.ReadImage(brain_mask_file)).astype(bool)

    if isinstance(IC_maps, np.ndarray): # can take in a pre-loaded array as input
        if not volume_idx.sum()==IC_maps.shape[1]:
            raise ValueError("The input IC_maps array does not have the expected number of voxels in volume_idx.")
        ICs_arr = IC_maps
    elif Path(IC_maps).is_file():
        ICs_arr = load_img_to_array(IC_maps, brain_mask_file, input_is_mask=False)
    else:
        raise ValueError("IC_maps is neither an array nor a file.")

    if not IC_idx:
        IC_idx = list(range(ICs_arr.shape[0])) # take all ICs

    DR_dict = {}
    for img_id in img_id_list:
        DR_dict[img_id] = {}
        timeseries = timeseries_dict[img_id]
        DR = dual_regression(ICs_arr, timeseries)
        DR_maps = DR['C']*DR['S']
        DR_dict[img_id]['DR_maps']=DR_maps[:,IC_idx]
        DR_dict[img_id]['DR_amp']=DR['S'][IC_idx]
        DR_dict[img_id]['DR_time']=DR['W'][:,IC_idx]
        DR_corr = np.corrcoef(DR['W'][:,IC_idx].T)
        if z_corr:
            DR_corr = np.arctanh(DR_corr)
        if len(IC_idx)>1: # it raises an error is array is 1 row
            np.fill_diagonal(DR_corr, np.nan)
        else:
            DR_corr = np.nan
        DR_dict[img_id]['DR_cross_corr']=DR_corr
    return DR_dict


def compute_global_connectivity(timeseries_dict, img_id_list, verbose=True):
    """Compute global connectivity for each image ID within a timeseries_dict.

    Global connectivity is defined as the mean correlation of each voxel
    with all other voxels in the brain.

    Parameters
    ----------
    timeseries_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated timeseries array,
        e.g. :attr:`rubella.data_obj.DataObject.timeseries_dict`.

    img_id_list : list of str
        List of image IDs that are keys for ``timeseries_dict``,
        e.g. :attr:`rubella.data_obj.DataObject.img_id_list`.

    verbose : bool, default=True
        Whether to print the image ID for each iteration, which can be useful to track progress.

    Returns
    -------
    GC_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated global
        connectivity map.
    """

    GC_dict = {}
    for img_id in img_id_list:
        if verbose:
            print(img_id)
        cross_corr = np.corrcoef(timeseries_dict[img_id].T)
        cross_corr[np.isnan(cross_corr)] = 0
        GC_dict[img_id] = cross_corr.mean(axis=0)
        del cross_corr
    return GC_dict


def compute_FC_matrix(timeseries_dict, brain_mask_file, img_id_list, parcellation, roi_ids=None, z_corr=False, parcellated=False):
    """Compute a parcellated functional connectivity matrix for each image ID within a timeseries_dict.

    For each subject, the timeseries of each parcel is obtained by averaging across its constituent
    voxels, and a pairwise correlation matrix is then computed across all parcels.
    The diagonal is set to NaN.

    Parameters
    ----------
    timeseries_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated timeseries array,
        e.g. :attr:`rubella.data_obj.DataObject.timeseries_dict`.

    brain_mask_file : str or pathlib.Path
        The full path to the brain mask Nifti file,
        e.g. :attr:`rubella.data_obj.DataObject.brain_mask_file`.

    img_id_list : list of str
        List of image IDs that are keys for ``timeseries_dict``,
        e.g. :attr:`rubella.data_obj.DataObject.img_id_list`.

    parcellation : str, pathlib.Path, numpy.ndarray or list
        The input parcellation defining the ROIs. Accepted formats are: a path to a multi-label
        Nifti file, a pre-loaded 1D integer array, a list of single-label Nifti paths, or a list
        of pre-loaded binary 1D arrays.

    roi_ids : list, default=None
        Optionally, a list of labels to assign to each parcel. Must match the number of parcels.
        If None, integer labels are assigned automatically.

    z_corr : bool, default=False
        Whether to output Fisher z transformed correlation values, instead of Pearson's r.
        Note that the diagonal is set to NaN after the transform is applied.

    parcellated : bool, default=False
        Whether the input timeseries are already parcellated rather than being voxelwise.
        If ``parcellated=True``, each entry in ``timeseries_dict`` must have shape
        (n_timepoints, n_ROIs), and ``parcellation``, ``brain_mask_file`` and ``roi_ids`` are not
        used by the function. Moreover, ``'parcel_ids'`` is not an entry in the output dictionary.

    Returns
    -------
    FCM_dict : dict
        Nested dictionary with ``img_id_list`` as keys. Each entry is a dictionary with the
        following keys:

        ``'parcel_ids'`` : list
            The ordered list of parcel labels corresponding to the rows/columns of the matrix.
        ``'matrix'`` : numpy.ndarray of shape (n_parcels, n_parcels)
            The functional connectivity matrix for that subject. The diagonal is NaN.
    """
    if not parcellated:
        parcels_d = _parse_parcellation(parcellation, brain_mask_file, roi_ids=roi_ids)
        parcel_ids = list(parcels_d.keys())

    FCM_dict = {}
    for img_id in img_id_list:
        FCM_dict[img_id] = {}
        if parcellated:
            parcellated_timeseries = timeseries_dict[img_id].T
        else:
            FCM_dict[img_id]['parcel_ids'] = parcel_ids #i moved it here, else parcel_ids is not defined for parcellated=True
            parcellated_timeseries = np.array([timeseries_dict[img_id][:,parcels_d[p]].mean(axis=1) for p in parcel_ids])
        corr_matrix = np.corrcoef(parcellated_timeseries)
        if z_corr:
            corr_matrix = np.arctanh(corr_matrix)
        np.fill_diagonal(corr_matrix, np.nan)
        FCM_dict[img_id]['matrix'] = corr_matrix
    return FCM_dict
    

def get_concat_timeseries(timeseries_dict, img_id_list):
    concat_timeseries = timeseries_dict[img_id_list[0]]
    for img_id in img_id_list[1:]:
        concat_timeseries = np.concatenate((concat_timeseries, timeseries_dict[img_id]), axis=0)
    print(concat_timeseries.shape)
    return concat_timeseries


def get_sliding_windows(timeseries_dict, img_id_list, censoring_mask_dict=None, window_length=10, step_size=1, min_censored_window_length=1, fc_func=None, fc_kwargs={}):
    """Extract sliding windows of timeseries for each image ID within a timeseries_dict.

    Sliding windows support motion censoring, and optionally the computation of functional
    connectivity within each window.

    For each subject, windows are stepped across the full original temporal extent of
    the scan. When a censoring mask is provided, censored frames are excluded from each
    window's data, so a window may contain fewer rows than ``window_length``.
    Windows with fewer remaining frames than ``min_censored_window_length`` are replaced by None;
    setting ``min_censored_window_length=window_length`` thus only retains fully contiguous
    (uncensored) windows.

    If ``fc_func`` is provided, each window's timeseries is passed through that function
    and the output connectivity array is stored in place of the raw timeseries.
    The function must accept ``timeseries_dict`` and ``img_id_list`` as keyword arguments
    (following the package convention), which are injected automatically per window.
    All other required arguments should be passed via ``fc_kwargs``.

    Parameters
    ----------
    timeseries_dict : dict
        Dictionary with ``img_id_list`` as keys, and each entry is the associated timeseries array
        of shape (n_retained_frames, n_voxels), where n_retained_frames is the number of frames
        after censoring has been applied upstream,
        e.g. :attr:`rubella.data_obj.DataObject.timeseries_dict`.

    img_id_list : list of str
        Ordered list of image IDs to process. Must be keys of ``timeseries_dict``.

    censoring_mask_dict : dict, default=None
        Dictionary with image IDs as keys, and each entry is a boolean array of length
        n_total_frames (i.e. the original, pre-censoring frame count), e.g.
        :attr:`rubella.data_obj.DataObject.censoring_mask`. True indicates a frame was retained;
        False indicates it was censored. If an image ID is absent, no censoring is applied for
        that subject.

    window_length : int, default=10
        Number of original (pre-censoring) frames spanned by each window.

    step_size : int, default=1
        Number of frames to advance between successive windows. Must be >= 1.

    min_censored_window_length : int, default=1
        The minimal actual length of each window (after applying censoring). Any
        window that does not meet this threshold is replaced by None.
        If ``min_censored_window_length=window_length``, any window with 1 or more censored frames
        is replaced by None.

    fc_func : callable, default=None
        A functional connectivity function to apply to each window's timeseries, e.g.
        :func:`seed_based_FC`. Must follow the package convention of accepting
        ``timeseries_dict`` and ``img_id_list`` as keyword arguments. It should return a
        dictionary with image IDs as keys, with each value being the connectivity array for that
        subject's window. If None, raw timeseries windows are returned.

    fc_kwargs : dict, default={}
        Additional keyword arguments forwarded to ``fc_func`` (e.g. ``brain_mask_file``,
        ``seed_roi``, ``z_corr``). The ``timeseries_dict`` and ``img_id_list`` arguments are
        injected automatically and should not be included here.

    Returns
    -------
    sliding_window_dict : dict
        Dictionary with the following keys:

        ``'sliding_windows'`` : dict
            For each image ID, a list with one array per window (or None for a window that does
            not meet ``min_censored_window_length``). If ``fc_func`` is None, each array has shape
            (n_uncensored_frames_in_window, n_voxels). If ``fc_func`` is provided, each array is
            the connectivity output for that window, with shape determined by ``fc_func``.
        ``'sliding_windows_idx'`` : dict
            For each image ID, a list of 1D integer arrays containing the indices into the
            retained timeseries (rows of ``timeseries_dict`` values) used to build each window.
            Useful for traceability and cross-referencing with other per-frame metadata.
        ``'window_censoring_mask'`` : dict
            For each image ID, a list of boolean arrays of length ``window_length``, giving the
            censoring mask of the original frames spanned by each window.

    Examples
    --------
    Return raw timeseries windows (default behaviour):

    >>> sliding_window_dict = get_sliding_windows(
    ...     timeseries_dict=data_obj.timeseries_dict,
    ...     img_id_list=data_obj.img_id_list,
    ...     censoring_mask_dict=data_obj.censoring_mask,
    ...     window_length=360,   # 6 min at TR=1s
    ...     step_size=180,       # 3 min step
    ...     min_censored_window_length=1,
    ... )

    Compute seed-based connectivity within each window by passing ``fc_func``.
    Note that ``timeseries_dict`` and ``img_id_list`` are injected automatically
    and must not be included in ``fc_kwargs``:

    >>> sliding_window_dict = get_sliding_windows(
    ...     timeseries_dict=data_obj.timeseries_dict,
    ...     img_id_list=data_obj.img_id_list,
    ...     censoring_mask_dict=data_obj.censoring_mask,
    ...     window_length=360,
    ...     step_size=180,
    ...     min_censored_window_length=1,
    ...     fc_func=rubella.basic.seed_based_FC,
    ...     fc_kwargs={
    ...         'brain_mask_file': data_obj.brain_mask_file,
    ...         'seed_roi': my_seed_roi,
    ...         'z_corr': True,
    ...     },
    ... )
    """

    if step_size < 1:
        raise ValueError(f"Step size must be at least 1, got {step_size} instead.")

    if censoring_mask_dict is None:
        censoring_mask_dict = {}

    sliding_window_dict = {}
    sliding_window_dict['sliding_windows'] = {img_id: [] for img_id in img_id_list}
    sliding_window_dict['window_censoring_mask'] = {img_id: [] for img_id in img_id_list}
    sliding_window_dict['sliding_windows_idx'] = {img_id: [] for img_id in img_id_list}

    # Build per-subject index arrays once, outside the window loop
    subject_meta = {}
    for img_id in img_id_list:
        timeseries = timeseries_dict[img_id]
        if img_id in censoring_mask_dict:
            censoring_mask = censoring_mask_dict[img_id]
            nframes = len(censoring_mask)
            idx_arr = np.array([None] * nframes)
            idx_arr[censoring_mask] = np.arange(timeseries.shape[0])
            censoring = True
        else:
            if not censoring_mask_dict=={}:
                print(f'No censoring mask for {img_id}')
            nframes = timeseries.shape[0]
            idx_arr = np.arange(timeseries.shape[0])
            censoring_mask = np.ones(nframes, dtype=bool)
            censoring = False
        subject_meta[img_id] = (timeseries, censoring_mask, idx_arr, nframes, censoring)

    # Verify all subjects have the same scan length
    all_nframes = {img_id: subject_meta[img_id][3] for img_id in img_id_list}
    unique_lengths = set(all_nframes.values())
    if len(unique_lengths) > 1:
        lengths_str = "\n".join(f"  {img_id}: {n} frames" for img_id, n in all_nframes.items())
        import warnings
        warnings.warn(
            f"WARNING: Subjects have different scan lengths!\n"
            f"The window loop will run up to the longest scan ({max(unique_lengths)} frames). "
            f"Subjects with shorter scans will produce fewer windows at the end of the timeseries.\n"
            f"Subject frame counts:\n{lengths_str}\n",
            UserWarning, stacklevel=2
        )

    # Use the maximum scan length to drive the window loop, so that all timepoints
    # are covered. Subjects with shorter scans will naturally produce fewer windows
    # at the end as their timeseries runs out.
    nframes = max(subject_meta[img_id][3] for img_id in img_id_list)

    i = 0
    while i <= (nframes - window_length):
        # Build a single-window timeseries_dict for all subjects at position i
        window_ts_dict = {}
        window_idx_dict = {}

        img_id_include = []
        img_id_empty = []
        for img_id in img_id_list:
            timeseries, censoring_mask, idx_arr, _, censoring = subject_meta[img_id]
            window_censoring_mask = censoring_mask[i:i + window_length]
            sliding_window_dict['window_censoring_mask'][img_id].append(window_censoring_mask)

            if min_censored_window_length <= window_censoring_mask.sum(): # check if the minimum number of remaining frames is met
                window_idx = idx_arr[i:i + window_length][window_censoring_mask].astype(int)
                window_ts_dict[img_id] = timeseries[window_idx, :]
                window_idx_dict[img_id] = window_idx
                img_id_include.append(img_id)
            else:
                img_id_empty.append(img_id)

        if fc_func is not None:
            fc_output = fc_func(
                timeseries_dict=window_ts_dict,
                img_id_list=img_id_include,
                **fc_kwargs
            )

            # Validate that fc_output keys match the expected img_ids for this window
            expected_ids = set(window_ts_dict.keys())
            returned_ids = set(fc_output.keys())
            if expected_ids != returned_ids:
                missing = expected_ids - returned_ids
                unexpected = returned_ids - expected_ids
                raise ValueError(
                    f"fc_func '{fc_func.__name__}' returned unexpected keys at window position i={i}. "
                    f"Missing img_ids: {missing if missing else 'none'}. "
                    f"Unexpected img_ids: {unexpected if unexpected else 'none'}."
                )

            for img_id, fc_arr in fc_output.items():
                sliding_window_dict['sliding_windows'][img_id].append(fc_arr)
                sliding_window_dict['sliding_windows_idx'][img_id].append(window_idx_dict[img_id])
        else:
            for img_id, window_ts in window_ts_dict.items():
                sliding_window_dict['sliding_windows'][img_id].append(window_ts)
                sliding_window_dict['sliding_windows_idx'][img_id].append(window_idx_dict[img_id])

        # also fill the empty slots so that there is always an entry for each window
        for img_id in img_id_empty:
            sliding_window_dict['sliding_windows'][img_id].append(None)
            sliding_window_dict['sliding_windows_idx'][img_id].append(None)

        i += step_size

    return sliding_window_dict


'''
Utils
'''

def _parse_parcellation(parcellation, brain_mask_file, roi_ids=None):
    """
    Parse parcellation input into a dictionary of {label_id: 1D bool mask}.

    Accepted inputs:
        - str/Path          : single multi-label NIfTI → one mask per unique label
        - np.ndarray        : pre-loaded multi-label 1D array (brain voxels only)
        - list[str/Path]    : one single-label NIfTI per seed
        - list[np.ndarray]  : one pre-loaded 1D array per seed

    Parameters
    ----------
    parcellation    : see above
    brain_mask_file : path to brain mask NIfTI
    roi_ids         : optional list of strings to use as dictionary keys.
                      Must match the number of labels/seeds in parcellation.

    Returns
    -------
    parcels : dict mapping label ID (int or str) → 1D bool np.ndarray

    Notes
    -----
    Generated with Claude Sonnet 4.6 (Anthropic), claude.ai — May 2026.    
    """

    brain_mask_file = Path(brain_mask_file)

    def _check_array(arr, idx=""):
        volume_idx = sitk.GetArrayFromImage(sitk.ReadImage(brain_mask_file)).astype(bool)
        if volume_idx.sum() != len(arr):
            raise ValueError(
                f"Array{f' at index {idx}' if idx != '' else ''} length {len(arr)} "
                f"does not match brain mask voxel count {volume_idx.sum()}."
            )

    def _check_integer_labels(arr):
        unique_vals = np.unique(arr)
        float_vals = unique_vals[unique_vals != np.floor(unique_vals)]
        if len(float_vals) > 0:
            raise ValueError(
                f"Parcellation array contains non-integer label values: {float_vals.tolist()}"
            )

    def _check_binary_mask(arr, idx=""):
        unique_vals = np.unique(arr)
        if not np.all(np.isin(unique_vals, [0, 1])):
            unexpected = unique_vals[~np.isin(unique_vals, [0, 1])]
            raise ValueError(
                f"Seed mask{f' at index {idx}' if idx != '' else ''} is not binary. "
                f"Unexpected values: {unexpected.tolist()}"
            )

    def _apply_ids(labels, masks):
        if roi_ids is not None:
            if len(roi_ids) != len(labels):
                raise ValueError(
                    f"roi_ids has {len(roi_ids)} entries but parcellation has {len(labels)} labels."
                )
            labels = roi_ids
        return dict(zip(labels, masks))

    # Single NIfTI path → multi-label
    if isinstance(parcellation, (str, os.PathLike)):
        arr = load_img_to_array(parcellation, brain_mask_file, input_is_mask=True)
        _check_integer_labels(arr)
        unique_labels = np.unique(arr[arr != 0]).astype(int)
        return _apply_ids(unique_labels.tolist(), [arr == l for l in unique_labels])

    # Pre-loaded array → multi-label
    if isinstance(parcellation, np.ndarray):
        _check_array(parcellation)
        _check_integer_labels(parcellation)
        unique_labels = np.unique(parcellation[parcellation != 0]).astype(int)
        return _apply_ids(unique_labels.tolist(), [parcellation == l for l in unique_labels])

    # List inputs
    if isinstance(parcellation, list):
        if len(parcellation) == 0:
            raise ValueError("Parcellation list is empty.")

        if isinstance(parcellation[0], (str, os.PathLike)):
            masks = []
            for i, p in enumerate(parcellation):
                arr = load_img_to_array(p, brain_mask_file, input_is_mask=True)
                _check_binary_mask(arr, idx=i)
                masks.append(arr.astype(bool))
            return _apply_ids(list(range(1, len(masks) + 1)), masks)

        if isinstance(parcellation[0], np.ndarray):
            masks = []
            for i, arr in enumerate(parcellation):
                _check_array(arr, idx=i)
                _check_binary_mask(arr, idx=i)
                masks.append(arr.astype(bool))
            return _apply_ids(list(range(1, len(masks) + 1)), masks)

    raise TypeError(
        f"Unsupported parcellation type: {type(parcellation).__name__}. "
        "Expected a NIfTI path, np.ndarray, or list of either."
    )
