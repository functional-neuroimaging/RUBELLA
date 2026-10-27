"""
CORE FUNCTIONS FOR MANAGING IMAGE/DATA FORMATS USED ELSEWHERE
"""
import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path


def load_img_to_masked_array(input_img, brain_mask, data_type, resample_to_brain_mask=True, verbose=True):
    """Load an input image as an array, including only the voxels within the brain mask.

    The input can be provided as a Nifti file, a SimpleITK image or a nibabel image, and
    ``data_type`` determines how the input is resampled (if needed) and which assumptions about
    its values are checked:

    * ``'mask'``: a 3D binary mask (e.g. a seed region), with values of only 0 and 1.
    * ``'labels'``: a 3D parcellation, with integer values (0 is the background).
    * ``'continuous'``: quantitative data, either a 3D map or a 4D timeseries.

    The brain mask determines the expected image grid, i.e. the array size, voxel spacing, origin
    and orientation. If the input image grid does not match the brain mask, the image is resampled
    onto the brain mask grid before being loaded as an array (with ``resample_to_brain_mask=True``
    by default). IMPORTANTLY: resampling does not register the images, and assumes that the two
    images are already aligned in physical space - if not, the loaded array is likely to be
    misaligned or empty after resampling.

    Parameters
    ----------
    input_img : str, pathlib.Path, SimpleITK.Image or nibabel.Nifti1Image
        The input image, either as the path to a Nifti file or a pre-loaded image.

    brain_mask : str, pathlib.Path, SimpleITK.Image or nibabel.Nifti1Image
        The brain mask, which is used to determine brain voxel array indices. Either the path
        to a Nifti file or a pre-loaded image.

    data_type : str
        The type of data contained in ``input_img``, among ``'mask'``, ``'labels'`` or
        ``'continuous'``.

    resample_to_brain_mask : bool, default=True
        If True, and the input image grid (array size, spacing, origin or orientation) does not
        match ``brain_mask``, the input image is resampled onto the image grid of the brain mask so the data can be loaded
        in a commonspace. Masks and labels are resampled with the antsApplyTransforms GenericLabel
        interpolator, and continuous data with linear interpolation. If False, any mismatch throws
        an error.

    verbose : bool, default=True
        Whether to print warnings, i.e. when resampling the input, or if continuous data contain
        NaN or infinite values.

    Returns
    -------
    img_array : numpy.ndarray
        The output image array, 1D of shape (n_voxels,) for a 3D input image, or 2D of shape
        (n_timepoints, n_voxels) for a 4D input image. The array is of type bool for
        ``data_type='mask'``, int for ``'labels'``, and float for ``'continuous'``
        (the float precision of the input image is kept, e.g. float32).
    """
    if data_type not in ['mask', 'labels', 'continuous']:
        raise ValueError(f"data_type must be 'mask', 'labels' or 'continuous', got {data_type} instead.")

    sitk_img = _read_sitk_image(input_img)
    mask_img = _read_sitk_image(brain_mask)
    volume_idx = sitk.GetArrayFromImage(mask_img).astype(bool)

    dim = sitk_img.GetDimension()
    if data_type in ['mask', 'labels'] and not dim==3:
        raise ValueError(f"An input of data_type='{data_type}' must be a 3D image, got a {dim}D image instead.")
    elif not dim in [3, 4]:
        raise ValueError(f"The input image must be 3D or 4D, got a {dim}D image instead.")

    # check the voxel values before resampling, since resampling can change them
    if data_type == 'mask':
        arr = sitk.GetArrayFromImage(sitk_img)
        if not np.isin(arr, [0, 1]).all():
            unexpected = np.unique(arr[~np.isin(arr, [0, 1])])
            raise ValueError(f"The input mask is not binary. Unexpected values: {unexpected.tolist()[:10]}")
    elif data_type == 'labels':
        arr = sitk.GetArrayFromImage(sitk_img)
        if not (arr == np.round(arr)).all():
            unexpected = np.unique(arr[arr != np.round(arr)])
            raise ValueError(f"The input labels contain non-integer values: {unexpected.tolist()[:10]}")

    grid_mismatches = _get_grid_mismatches(sitk_img, mask_img)
    if len(grid_mismatches)>0:
        warning_str="""
        Warning: The image grid of the input image does not match the brain mask (input vs brain mask):
"""+"\n".join(grid_mismatches)+"\n"

        if resample_to_brain_mask:
            warning_str+="""
        The image is resampled onto the brain mask grid, with the assumption that
        both images are already aligned in physical space.
        """
            if verbose:
                print(warning_str)
            if data_type in ['mask', 'labels']:
                from rabies.utils import antsApplyTransforms
                import tempfile, shutil
                # antsApplyTransforms reads its inputs from files
                tmppath = tempfile.mkdtemp()
                input_file = f"{tmppath}/input.nii.gz"
                ref_file = f"{tmppath}/brain_mask.nii.gz"
                out_file = f"{tmppath}/resampled_mask.nii.gz"
                sitk.WriteImage(sitk_img, input_file)
                sitk.WriteImage(mask_img, ref_file)
                antsApplyTransforms(transforms = [], inverses = [],
                    input_image = input_file, ref_image = ref_file, output_filename = out_file, interpolation='GenericLabel', rabies_data_type=sitk.sitkInt16, clip_negative=False)
                sitk_img = sitk.ReadImage(out_file, sitk.sitkInt16)
                shutil.rmtree(tmppath)
            else:
                from rabies.utils import resample_volumes
                sitk_img = resample_volumes(in_img=sitk_img, in_ref=mask_img, transforms_3d_files=[], inverses_3d=[],
                                                        motcorr_params_file=None, interpolation=sitk.sitkLinear, rabies_data_type=sitk.sitkFloat32, clip_negative=False)
        else:
            raise ValueError(warning_str)

    if dim==4:
        img_array = sitk.GetArrayFromImage(sitk_img)[:,volume_idx]
    else:
        img_array = sitk.GetArrayFromImage(sitk_img)[volume_idx]

    if data_type == 'mask':
        img_array = img_array.astype(bool)
        if not img_array.any():
            raise ValueError("The input mask has no voxels within the brain mask.")
    elif data_type == 'labels':
        img_array = img_array.astype(int)
        if not img_array.any():
            raise ValueError("The input labels have no labelled voxels within the brain mask.")
    else:
        if not np.issubdtype(img_array.dtype, np.floating): # keep the float precision of the input, e.g. float32 timeseries
            img_array = img_array.astype(float)
        if verbose and not np.isfinite(img_array).all():
            print(f"Warning: the input image contains {(~np.isfinite(img_array)).sum()} NaN or infinite values within the brain mask.")
    return img_array


def _get_grid_mismatches(sitk_img, mask_img):
    """Compare the 3D image grid (size, spacing, origin and orientation) of an image with the brain mask.

    Returns a list of strings describing each mismatch, which is empty if the grids match.
    Small differences, e.g. from the float precision of Nifti headers, are tolerated.
    """
    dim = sitk_img.GetDimension()
    input_direction = np.array(sitk_img.GetDirection()).reshape(dim, dim)[:3,:3] # only keep the 3 spatial axes
    mask_direction = np.array(mask_img.GetDirection()).reshape(3, 3)
    mask_spacing = np.array(mask_img.GetSpacing())

    mismatches = []
    if not sitk_img.GetSize()[:3]==mask_img.GetSize():
        mismatches.append(f"        size: {sitk_img.GetSize()[:3]} vs {mask_img.GetSize()}")
    if not np.allclose(sitk_img.GetSpacing()[:3], mask_spacing, rtol=1e-4, atol=0):
        mismatches.append(f"        spacing: {sitk_img.GetSpacing()[:3]} vs {mask_img.GetSpacing()}")
    # the origin can differ by a small fraction of a voxel
    if not np.allclose(sitk_img.GetOrigin()[:3], mask_img.GetOrigin(), rtol=0, atol=1e-3*mask_spacing.min()):
        mismatches.append(f"        origin: {sitk_img.GetOrigin()[:3]} vs {mask_img.GetOrigin()}")
    if not np.allclose(input_direction, mask_direction, rtol=0, atol=1e-4):
        mismatches.append(f"        orientation: {input_direction.flatten().tolist()} vs {mask_direction.flatten().tolist()}")
    return mismatches


def _read_sitk_image(img):
    """Return a SimpleITK image from a Nifti file path, a SimpleITK image or a nibabel image."""
    if isinstance(img, sitk.Image):
        return img
    if isinstance(img, (str, os.PathLike)):
        if not Path(img).is_file():
            raise FileNotFoundError(f"The input file {img} does not exist.")
        return sitk.ReadImage(Path(img))

    import nibabel as nib
    if isinstance(img, nib.Nifti1Image): # also includes nib.Nifti2Image
        import tempfile, shutil
        # convert through a file, so that the image header is read exactly as for a file input
        tmppath = tempfile.mkdtemp()
        tmp_file = f"{tmppath}/img.nii.gz"
        nib.save(img, tmp_file)
        sitk_img = sitk.ReadImage(tmp_file)
        shutil.rmtree(tmppath)
        return sitk_img

    raise TypeError(f"Unsupported image type: {type(img).__name__}. "
                    "Expected a Nifti file path, a SimpleITK image or a nibabel image.")


def get_img_from_array(img_array, brain_mask_file, image_type='SITK'):
    """Recover an image from an array of brain voxels, i.e. the inverse of :func:`load_img_to_masked_array`.

    A 1D input array is converted into a 3D volume, and a 2D input array (time by voxel) is
    converted into a 4D image. Each voxel value is placed back at its location within the brain
    mask, and voxels outside the brain mask are set to 0. The output image takes the voxel
    spacing, origin and orientation of the brain mask (for a 4D image, the spacing along time is 1).

    This function can be used to save outputs to a .nii.gz Nifti file, for instance a brain map
    derived from :func:`rubella.basic_fc.seed_based_FC` (see the Examples below).

    Parameters
    ----------
    img_array : numpy.ndarray
        The input array, either 1D of shape (n_voxels,) for a 3D volume, or 2D of shape
        (n_timepoints, n_voxels) for a 4D image. The voxel indices must match the brain voxels
        of ``brain_mask_file``, as for arrays loaded with :func:`load_img_to_masked_array`.

    brain_mask_file : str or pathlib.Path
        Path to the brain mask Nifti file, which is used to determine brain voxel array indices.

    image_type : str, default='SITK'
        The format of the output image, either ``'SITK'`` for a SimpleITK image, or
        ``'nibabel'`` for a nibabel image.

    Returns
    -------
    img : SimpleITK.Image or nibabel.Nifti1Image
        The output image, in the format determined by ``image_type``.

    Examples
    --------
    Save a brain map to a Nifti file as a SimpleITK image:

    >>> import SimpleITK as sitk
    >>> img = get_img_from_array(SBC_map, data_obj.brain_mask_file, image_type='SITK')
    >>> sitk.WriteImage(img, 'SBC_map.nii.gz')

    Or as a nibabel image:

    >>> import nibabel as nib
    >>> img = get_img_from_array(SBC_map, data_obj.brain_mask_file, image_type='nibabel')
    >>> nib.save(img, 'SBC_map.nii.gz')
    """
    if image_type not in ['SITK', 'nibabel']:
        raise ValueError(f"image_type must be 'SITK' or 'nibabel', got {image_type} instead.")

    img_array = np.asarray(img_array)
    if img_array.dtype == bool:
        img_array = img_array.astype(np.uint8)  # boolean arrays are not supported by image formats

    mask_img = sitk.ReadImage(Path(brain_mask_file))
    volume_idx = sitk.GetArrayFromImage(mask_img).astype(bool)
    num_voxels = volume_idx.sum()

    if not img_array.ndim in [1, 2]:
        raise ValueError(f"img_array must be 1D or 2D, got an array of {img_array.ndim} dimensions instead.")
    if not img_array.shape[-1] == num_voxels:
        raise ValueError(f"img_array has {img_array.shape[-1]} voxels, but the brain mask has {num_voxels} voxels.")

    # fill the brain voxels of an empty volume (or timeseries of volumes) with the input values
    if img_array.ndim == 1:
        full_array = np.zeros(volume_idx.shape, dtype=img_array.dtype)
        full_array[volume_idx] = img_array
    else:
        full_array = np.zeros((img_array.shape[0],) + volume_idx.shape, dtype=img_array.dtype)
        full_array[:, volume_idx] = img_array

    if image_type == 'nibabel':
        import nibabel as nib
        # nibabel orders array axes as (x, y, z, time), the reverse of SimpleITK arrays
        return nib.Nifti1Image(full_array.T, nib.load(Path(brain_mask_file)).affine)

    img = sitk.GetImageFromArray(full_array, isVector=False)
    if img_array.ndim == 1:
        img.CopyInformation(mask_img)
    else:
        # extend the 3D spatial information of the brain mask with a 4th time dimension
        img.SetSpacing(mask_img.GetSpacing() + (1.0,))
        img.SetOrigin(mask_img.GetOrigin() + (0.0,))
        direction = np.eye(4)
        direction[:3, :3] = np.array(mask_img.GetDirection()).reshape(3, 3)
        img.SetDirection(direction.flatten())
    return img


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
    """Parse a parcellation input into a dictionary of ROI masks.

    Parameters
    ----------
    parcellation : str, pathlib.Path, SimpleITK.Image, nibabel.Nifti1Image, numpy.ndarray or list
        Either a single multi-label input, giving one ROI per unique non-zero label, or a list of
        binary masks, giving one ROI per mask. Each input can be the path to a Nifti file, a
        pre-loaded image, or a pre-loaded 1D array of shape (n_voxels,).

    brain_mask_file : str or pathlib.Path
        Path to the brain mask Nifti file, which is used to determine brain voxel array indices.

    roi_ids : list, default=None
        Optionally, a list of IDs to use as dictionary keys. Must match the number of ROIs.
        If None, the label values are used for a multi-label input, and 1 to n_ROIs for a list
        of masks.

    Returns
    -------
    parcels : dict
        Dictionary mapping each ROI ID to a 1D boolean array of shape (n_voxels,).
    """

    def _load(input_img, data_type):
        # arrays are converted to an image, so that all inputs go through the same checks when loading
        if isinstance(input_img, np.ndarray):
            input_img = get_img_from_array(input_img, brain_mask_file)
        return load_img_to_masked_array(input_img, brain_mask_file, data_type=data_type)

    if isinstance(parcellation, list):
        if len(parcellation) == 0:
            raise ValueError("Parcellation list is empty.")
        masks = [_load(p, 'mask') for p in parcellation]
        labels = list(range(1, len(masks) + 1))
    else:
        arr = _load(parcellation, 'labels')
        labels = np.unique(arr[arr != 0]).tolist()
        masks = [arr == l for l in labels]

    if roi_ids is not None:
        if len(roi_ids) != len(labels):
            raise ValueError(f"roi_ids has {len(roi_ids)} entries but parcellation has {len(labels)} ROIs.")
        labels = roi_ids
    return dict(zip(labels, masks))
