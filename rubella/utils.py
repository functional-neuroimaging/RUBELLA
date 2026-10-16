"""
CORE FUNCTIONS FOR MANAGING IMAGE/DATA FORMATS USED ELSEWHERE
"""
import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path


def load_img_to_array(img_file, brain_mask_file, input_is_mask=False, resample_to_brain_mask=True):
    """Load a Nifti image as an array, including only the voxels within the brain mask.

    The input brain mask determines the expected image array spatial dimensions and resolution.
    If the input ``img_file`` does not match the resolution of the brain mask, the image
    is resampled to match the brain mask volume before being loaded as an array (with 
    ``resample_to_brain_mask=True`` by default).
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

    resample_to_brain_mask : bool, default=True
        If True, and the input image array size does not match the array size of the ``brain_mask_file``,
        the input image is resampled onto the image grid of the brain mask so the data can be loaded
        in a commonspace. If False, any mismatch throws an error.
        
    Returns
    -------
    img_array : numpy.ndarray
        The output image array, 2D (time by voxel) for a 4D input image, or 1D for a 3D image.
    """
    img_file = Path(img_file)
    brain_mask_file = Path(brain_mask_file)
    sitk_img = sitk.ReadImage(img_file)

    mask_img = sitk.ReadImage(brain_mask_file)
    volume_idx = sitk.GetArrayFromImage(mask_img).astype(bool)

    input_size=sitk_img.GetSize()
    mask_size=mask_img.GetSize()
    if not input_size[:3]==mask_size:
        warning_str=f"""
        Warning: The input image array of size {input_size} does not match the 
        brain mask array of size {mask_size}.
        """

        if resample_to_brain_mask:
            warning_str+="""
        The image is resampled with the assumption that it occupies 
        the same space as the brain mask.
        """
            print(warning_str)
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
        else:
            raise ValueError(warning_str)

    dim = sitk_img.GetDimension()
    if dim==4:
        img_array = sitk.GetArrayFromImage(sitk_img)[:,volume_idx]
    elif dim==3:
        img_array = sitk.GetArrayFromImage(sitk_img)[volume_idx]
    return img_array


def get_concat_timeseries(timeseries_dict, img_id_list):
    concat_timeseries = timeseries_dict[img_id_list[0]]
    for img_id in img_id_list[1:]:
        concat_timeseries = np.concatenate((concat_timeseries, timeseries_dict[img_id]), axis=0)
    print(concat_timeseries.shape)
    return concat_timeseries


'''
Internal operations
'''

def _vcorrcoef(X, Y):  # return a correlation between each row of X with each row of Y
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = (X-Xm).dot((Y-Ym).T)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1).reshape(-1,1)*np.sum((Y-Ym)**2, axis=1).reshape(1,-1))
    r = r_num/r_den
    return r


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
