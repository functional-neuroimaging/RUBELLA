import SimpleITK as sitk
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import load_img_to_array

class DataObject:
    """Group-level container of timeseries images in a common 2D time-by-voxel format.

    The object loads a list of timeseries images into a common 2D time-by-voxel array format,
    which streamlines downstream analyses at the group level. The timeseries must already be
    in a commonspace overlapping with the input ``brain_mask_file``. The brain mask defines which
    voxels are loaded into the 2D time-by-voxel array.

    When reading each input timeseries file, it is assigned a unique image ID (either based on
    the filename or a custom input, see the ``img_id_list`` and ``fname_split`` parameters) which
    allows to retain the identity of each input within the :class:`DataObject`. The ID is used to
    store timeseries in dictionary format, the :attr:`timeseries_dict` attribute, and to retrieve
    a given image array. The full list of IDs is stored in the :attr:`img_id_list` attribute.

    When creating the object, a ``brain_mask_file`` must be provided at minimum. If
    ``timeseries_file_list`` is provided, timeseries are also loaded with
    :meth:`load_commonspace_data_dict`. Similarly, censoring masks are loaded with
    :meth:`load_censoring_mask` if ``censoring_file_list`` is provided.

    Parameters
    ----------
    brain_mask_file : str or pathlib.Path
        Full path to the brain mask Nifti file that will be used to load commonspace timeseries.

    timeseries_file_list : list of str or pathlib.Path, default=None
        A list of file paths, each corresponding to a Nifti 4D timeseries image.
        All images must be in the same commonspace as ``brain_mask_file``.

    img_id_list : list of str, default=None
        Optionally, a list of manually-defined IDs can be provided. Each entry in the list will be
        matched with the corresponding index in ``timeseries_file_list``.
        If None, the image ID is derived from the Nifti file name, as determined by the
        ``fname_split`` parameter.

    fname_split : str, default=None
        If ``img_id_list=None``, meaning that the image ID is a filename, the ID is cropped by
        taking the prefix before ``fname_split``. This is useful to remove the filename ending
        from the ID, which may otherwise prevent associating the ID to other file types, like a
        censoring mask file as described in :meth:`load_censoring_mask`.

    censoring_file_list : list of str or pathlib.Path, default=None
        A list of file paths, each corresponding to a .csv censoring mask file output from RABIES.
        Each file in the list is paired with a pre-existing image ID in :attr:`img_id_list`,
        by searching which ID is contained within the file name (e.g.
        ``'sub-001_ses-2_censoring_mask.csv'`` will be paired to the ID ``'sub-001_ses-2'``).
        Hence, it is important to set :attr:`img_id_list` when loading timeseries so that it can
        match those files.

    verbose : bool, default=True
        Print filenames during loading.

    Attributes
    ----------
    brain_mask_file : pathlib.Path
        A copy of the input ``brain_mask_file``.

    volume_coords : numpy.ndarray
        A 3D array with the same shape as the brain mask array, where voxels outside the brain
        mask are NaN, and each voxel inside the mask is providing the index value for where 
        this voxel can be accessed inside the arrays loaded with :func:`rubella.utils.load_img_to_array` 
        (e.g. the columns of each timeseries in :attr:`timeseries_dict`).

    img_id_list : list of str
        The image IDs loaded in the DataObject. These are also the keys to access values inside
        the dictionaries from the other attributes below.

    timeseries_dict : dict
        The loaded 2D timeseries (time by voxel) for each image, with keys corresponding to the
        image IDs.

    censoring_mask : dict
        The censoring mask arrays, with keys corresponding to the image IDs.

    grouping : dict
        The group ID attributed to each image ID, with keys corresponding to the image IDs.
    """

    def __init__(self, brain_mask_file, timeseries_file_list=None, img_id_list=None, fname_split=None, censoring_file_list=None, verbose=True):
        # the brain mask file defines the voxel commonspace to load each volume
        self.brain_mask_file = Path(brain_mask_file)

        # for each brain voxel, its index in the 1D vector format of loaded images; NaN outside the brain
        volume_idx = sitk.GetArrayFromImage(sitk.ReadImage(self.brain_mask_file)).astype(bool)
        self.volume_coords = np.full(volume_idx.shape, np.nan)
        self.volume_coords[volume_idx] = np.arange(volume_idx.sum())

        if timeseries_file_list:
            self.load_commonspace_data_dict(timeseries_file_list, img_id_list=img_id_list, fname_split=fname_split, verbose=verbose)

        if censoring_file_list:
            self.load_censoring_mask(censoring_file_list)

    def load_commonspace_data_dict(self, timeseries_file_list, img_id_list=None, fname_split=None, verbose=True):
        """Load a list of timeseries images into a dictionary of 2D arrays.

        The loaded timeseries are stored in the :attr:`timeseries_dict` attribute, and their
        corresponding image IDs in the :attr:`img_id_list` attribute. Any previously loaded
        timeseries are replaced.

        Parameters
        ----------
        timeseries_file_list : list of str or pathlib.Path
            A list of file paths, each corresponding to a Nifti 4D timeseries image.
            All images must be in the same commonspace as ``brain_mask_file``.

        img_id_list : list of str, default=None
            Optionally, a list of manually-defined IDs can be provided. Each entry in the list will be
            matched with the corresponding index in ``timeseries_file_list``.
            If None, the image ID is derived from the Nifti file name, as determined by the
            ``fname_split`` parameter.

        fname_split : str, default=None
            If ``img_id_list=None``, meaning that the image ID is a filename, the ID is cropped by
            taking the prefix before ``fname_split``. This is useful to remove the filename ending
            from the ID, which may otherwise prevent associating the ID to other file types, like a
            censoring mask file as described in :meth:`load_censoring_mask`.

        verbose : bool, default=True
            Print filenames during loading.

        Returns
        -------
        self : object
            The DataObject.
        """

        # all inputs are assumed to overlap in a commonspace
        timeseries_dict = {}
        num_imgs = len(timeseries_file_list)
        if not img_id_list:
            img_id_list = []
            new_list = True
        else:
            new_list = False
            if not len(img_id_list)==num_imgs:
                raise ValueError("The input img_id_list does not have the same length as timeseries_file_list.")

        for i in range(num_imgs):
            file = Path(timeseries_file_list[i])
            if new_list:
                img_id = pathlib.Path(file).name
                if fname_split:
                    img_id = img_id.rsplit(fname_split)[0]
                img_id_list.append(img_id)
            else:
                img_id = img_id_list[i]
            if img_id in list(timeseries_dict.keys()):
                raise ValueError(f'Duplicate found for {img_id}')
            if verbose:
                print(file)

            timeseries_dict[img_id] = self._load_img_to_array(
                file, input_is_mask=False, resample_to_brain_mask=False)

        self.img_id_list = img_id_list
        self.timeseries_dict = timeseries_dict
        return self

    def load_censoring_mask(self, censoring_file_list):
        """Assign a censoring mask array to each image ID.

        The censoring masks allow downstream functions to identify which frames were masked
        from the original timeseries (i.e. where there are 'holes' in the time domain).
        Each file in the list is paired with a pre-existing image ID in :attr:`img_id_list`,
        by searching which ID is contained within the file name (e.g.
        ``'sub-001_ses-2_censoring_mask.csv'`` will be paired to the ID ``'sub-001_ses-2'``).
        Hence, it is important to set :attr:`img_id_list` when loading timeseries so that it can
        match those files.

        The censoring masks are stored in the :attr:`censoring_mask` attribute.

        Parameters
        ----------
        censoring_file_list : list of str or pathlib.Path
            A list of file paths, each corresponding to a .csv censoring mask file output from RABIES.

        Returns
        -------
        self : object
            The DataObject.
        """

        if not hasattr(self, "censoring_mask"):
            self.censoring_mask = {}

        for img_id in self.img_id_list:
            # first remove any previous mask assignment
            if img_id in self.censoring_mask.keys():
                print(f"Removing old censoring mask entry for {img_id}")
                del self.censoring_mask[img_id]

            for censoring_file in censoring_file_list:
                censoring_file = Path(censoring_file)
                if img_id in censoring_file.name:
                    if img_id in self.censoring_mask.keys():
                        raise ValueError(f"{img_id} is assigned to multiple censoring mask files - review the inputs and fname_split.")
                    self.censoring_mask[img_id]=pd.read_csv(censoring_file)['False = Masked Frames'].values

            if not img_id in self.censoring_mask.keys():
                print(f'No censoring mask was assigned to {img_id}.') 
        return self

    def assign_group(self, group_str_list):
        """Assign a group ID to each image ID.

        You must provide a list of strings, each defining a distinct group ID for your dataset.
        The function assumes that a single group ID is found inside the image ID string, in order
        to carry the assignment (hence you must make sure :attr:`img_id_list` is set so that the
        group IDs can be retrieved from the ID name).

        The group assignments are stored in the :attr:`grouping` attribute.

        Parameters
        ----------
        group_str_list : list of str
            A list of strings, where each entry is a possible group ID.

        Returns
        -------
        self : object
            The DataObject.
        """

        if not hasattr(self, "grouping"):
            self.grouping = {}
        for img_id in self.img_id_list:
            # first remove any previous group assignment
            if img_id in self.grouping.keys():
                print(f"Removing old group entry {self.grouping[img_id]} for {img_id}")
                del self.grouping[img_id]
            for group_str in group_str_list:
                if group_str in img_id:
                    if img_id in self.grouping.keys():
                        raise ValueError(f"{img_id} is assigned to multiple groups {self.grouping[img_id]} and {group_str}.")
                    self.grouping[img_id]=group_str
            if not img_id in self.grouping.keys():
                print(f'No group was assigned to {img_id}.')
        return self

    def _load_img_to_array(self, img_file, input_is_mask=False, resample_to_brain_mask=True):
        """Load an input Nifti image to the 2D array format of the DataObject.

        Wrapper for :func:`rubella.utils.load_img_to_array`, using the DataObject's
        ``brain_mask_file``. See that function for more details.

        Parameters
        ----------
        img_file : str or pathlib.Path
            Path to the input Nifti file.

        input_is_mask : bool, default=False
            Whether the input ``img_file`` is itself a brain mask or parcellation, in which case
            the resampling is carried using the antsApplyTransforms GenericLabel interpolator.

        resample_to_brain_mask : bool, default=True
            If True, and the input image array size does not match the array size of the ``brain_mask_file``,
            the input image is resampled onto the image grid of the brain mask so the data can be loaded
            in a commonspace. If False, any mismatch throws an error.
            
        Returns
        -------
        img_array : numpy.ndarray
            The output image array, 2D (time by voxel) for a 4D input image, or 1D for a 3D image.
        """
        return load_img_to_array(img_file,self.brain_mask_file, input_is_mask=input_is_mask, 
                                 resample_to_brain_mask=resample_to_brain_mask)
