import SimpleITK as sitk
import pathlib
import pandas as pd

class DataObject:
    """RUBELLA's DataObject.

    The object loads data from a set of timeseries images into a common 2D timeXspace array format,
    enabling common space-time matrix computations across the entire loaded dataset.

    Parameters
    ----------
    brain_mask_file : str
        Full path to the brain mask Nifti file that will be used to load commonspace timeseries.

    timeseries_file_list : list
        This must be a list of file paths, each corresponding to a Nifti 4D timeseries image. 
        All images must be in the same commonspace as brain_mask_file.

    img_id_list : list, default=None
        Optionally, a list of manually-defined IDs can be provided. Each entry in the list will be
        matched with the corresponding index in timeseries_file_list.
        If None is provided, the image ID will be the original Nifti file name.

    fname_split : str, default=None
        If img_id_list=None, meaning that img_id is a filename, the img_id will be cropped by taking 
        the prefix before fname_split. This is useful to remove the filename ending from img_id,
        which may prevent correctly associating the img_id to other file types, like a censoring mask
        file as described with load_censoring_mask() below.

    censoring_file_list : list, default=None
        A list of file paths, each corresponding to a .csv censoring mask file outputed from RABIES. 
        
    verbose : bool, default=True
        Print filenames during loading.

    Attributes
    ----------
    img_id_list : list
        The list of keys in the data dictionaries that correspond to each image ID.

    timeseries_dict : dictionary
        This dictionary contains the loaded 2D timeseries for each image, where keys correspond
        to the image IDs.

    num_timeframes : dictionary
        Number of timeframes for each image.
        
    censoring_mask : dictionary
        A dictionary of the censoring mask arrays for each img_id.

    """

    def __init__(self, timeseries_file_list, brain_mask_file, img_id_list=None, fname_split=None, censoring_file_list=None, verbose=True):
        # the brain mask file defines the voxel commonspace to load each volume
        self.brain_mask_file = brain_mask_file
        mask_img = sitk.ReadImage(brain_mask_file)
        volume_idx = sitk.GetArrayFromImage(mask_img).astype(bool)
        self.volume_idx = volume_idx

        # at minimum the data object loads a set of commonspace timeseries
        timeseries_dict, num_timeframes_dict, img_id_list = self.load_commonspace_data_dict(timeseries_file_list, img_id_list=img_id_list, fname_split=fname_split, verbose=verbose)
        self.img_id_list = img_id_list
        self.timeseries_dict = timeseries_dict
        self.num_timeframes = num_timeframes_dict

        if censoring_file_list:
            self.load_censoring_mask(censoring_file_list)

    def load_commonspace_data_dict(self, timeseries_file_list, img_id_list=None, fname_split=None, verbose=True):
        """Loads timeseries_file_list into a dictionary of 2D arrays.

        Parameters
        ----------
        timeseries_file_list : list
            This must be a list of file paths, each corresponding to a Nifti 4D timeseries image. 
            All images must be in the same commonspace as brain_mask_file.

        img_id_list : list, default=None
            Optionally, a list of manually-defined IDs can be provided. Each entry in the list will be
            matched with the corresponding index in timeseries_file_list.
            If None is provided, the image ID will be the original Nifti file name.

        fname_split : str, default=None
            If img_id_list=None, meaning that img_id is a filename, the img_id will be cropped by taking 
            the prefix before fname_split. This is useful to remove the filename ending from img_id,
            which may prevent correctly associating the img_id to other file types, like a censoring mask
            file as described with load_censoring_mask() below.

        verbose : bool, default=True
            Print filenames during loading.

        Returns
        -------
        img_id_list : list
            The list of keys in the data dictionaries that correspond to each image ID.
        timeseries_dict : dictionary
            This dictionary contains the loaded 2D timeseries for each image, where keys correspond
            to the image IDs.
        num_timeframes : dictionary
            Number of timeframes for each image.
        """

        # all inputs are assumed to overlap in a commonspace
        timeseries_dict = {}
        num_timeframes_dict = {}
        num_imgs = len(timeseries_file_list)
        if not img_id_list:
            img_id_list = []
            new_list = True
        else:
            new_list = False
            if not len(img_id_list)==num_imgs:
                raise ValueError("The input img_id_list does not have the same length as timeseries_file_list.")
        volume_idx = self.volume_idx

        for i in range(num_imgs):
            file = timeseries_file_list[i]
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
            timeseries = sitk.GetArrayFromImage(sitk.ReadImage(file))[:,volume_idx]
            timeseries_dict[img_id] = timeseries
            num_timeframes_dict[img_id] = timeseries.shape[0]

        return timeseries_dict, num_timeframes_dict, img_id_list

    def load_censoring_mask(self, censoring_file_list):
        """
        Assign a censoring mask array to each img_id, to correctly identify censored frames in 
        downstream analyses.
        Each file in censoring_file_list is paired by searching which img_id is contained within its
        file name (e.g. 'sub-001_ses-2' will be paired to img_id 'sub-001_ses-2').

        Parameters
        ----------
        censoring_file_list : list
            A list of file paths, each corresponding to a .csv censoring mask file outputed from RABIES. 
        
        Attributes
        ----------
        censoring_mask : dictionary
            A dictionary of the censoring mask arrays for each img_id.
            
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
                if img_id in censoring_file:
                    if img_id in self.censoring_mask.keys():
                        raise ValueError(f"{img_id} is assigned to multiple censoring mask files - review the inputs and img_id_split.")
                    self.censoring_mask[img_id]=pd.read_csv(censoring_file)['False = Masked Frames'].values

            if not img_id in self.censoring_mask.keys():
                print(f'No censoring mask was assigned to {img_id}.') 
        return self

    def assign_group(self, group_str_list):
        """
        Assign a group ID to each img_id. The group ID must be contained within img_id
        for automated assignment.

        Parameters
        ----------
        group_str_list : list
            A list of str, where each entry is a possible group ID.
        
        Attributes
        ----------
        grouping : dictionary
            A dictionary of the group ID for each img_id.
            
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
