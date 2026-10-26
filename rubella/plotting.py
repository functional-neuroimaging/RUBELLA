import matplotlib.pyplot as plt
import nilearn.plotting # this is imported so that 'cold_hot' colormap is available
from rabies.visualization import otsu_scaling
from rabies.analysis_pkg.diagnosis_pkg.dataset_QC import threshold_top_percent
from .utils import get_img_from_array
import SimpleITK as sitk
import numpy as np

def plot_brain(data,brain_mask_file,template_file=None,
               fig=None,ax=None,plot_size_scale=0.1,
               plane='coronal', slice_selection=6, slices_per_row=None, # slice selection
               cmap='cold_hot', vmax=None, vmin=None, alpha=1, plot_cbar=False, # cmap range
               fixed_threshold=None,percent_thresholding=False,top_percent=10, # image thresholding
               ):
    """Plot a brain map as a series of slices along one of the three cardinal planes.

    The brain map can be overlaid on top of a template image in the same space.

    Parameters
    ----------
    data : numpy.ndarray
        The 1D array of shape (n_voxels,) representing the brain map. The array indices must
        have been previously generated using ``brain_mask_file``.

    brain_mask_file : str or pathlib.Path
        Full path to the brain mask Nifti file.

    template_file : str or pathlib.Path, default=None
        Full path to the background template image. If None, no background is plotted.

    fig : matplotlib.figure.Figure, default=None
        A matplotlib figure. If None, the figure is automatically generated
        based on the finalized brain map plot dimensions.

    ax : matplotlib.axes.Axes, default=None
        The axis of ``fig`` to plot on. If None, ``fig`` must also be None.

    plot_size_scale : float, default=0.1
        A scaling factor for controlling the figure size when ``fig=None``.

    plane : str, default='coronal'
        The image plane among ``'sagittal'``, ``'coronal'``, or ``'horizontal'``.

    slice_selection : list of int, int or str, default=6
        The slices to plot along ``plane``. If a list of integers, these are the slice
        indices to plot. If an int, that number of slices is plotted at evenly spaced
        intervals across the brain. If ``'all'``, all slices containing brain voxels
        are plotted.

    slices_per_row : int, default=None
        The number of slices to plot in one row within the plot.
        If None, all slices are plotted in one row.

    cmap : str, default='cold_hot'
        Name of the matplotlib colormap.

    vmax : float, default=None
        The max value for the colormap. If None, it is set to the value of the 98th percentile
        of the voxel value distribution.

    vmin : float, default=None
        The minimum value for the colormap. If None, it is set to ``-vmax``.

    alpha : float or numpy.ndarray, default=1
        The transparency level of the brain map overlay. Must be between 0 and 1.
        It can be an array of the same shape as ``data`` to provide a per voxel alpha
        value.

    plot_cbar : bool, default=False
        Whether to plot a colorbar.

    fixed_threshold : float, default=None
        Voxels with an absolute value below this threshold are hidden in the plot.

    percent_thresholding : bool, default=False
        Whether to automatically threshold the image to only display the top X% of voxels
        with the highest absolute value. The percentage is set by ``top_percent``.

    top_percent : float, default=10
        The percentage value for ``percent_thresholding``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.

    ax : matplotlib.axes.Axes
        The axis containing the plot.

    cbar : matplotlib.colorbar.Colorbar or None
        The colorbar, or None if ``plot_cbar=False``.
    """

    # prepare slices to plot and load data image
    sitk_img = get_img_from_array(data, brain_mask_file, image_type='SITK')
    selected_slices = select_slices_for_plotting(slice_selection, brain_mask_file, plane)
    print(f'Plotting along slices: {selected_slices}')

    # select vmax at 98th percentile value
    vector = data.flatten()
    vector.sort()
    if vmax is None:
        vmax = vector[int(len(vector)*0.98)]
    if vmin is None:
        vmin=-vmax

    # prepare alpha as weighted array if needed
    weighted_alpha = False
    if isinstance(alpha, np.ndarray):
        if alpha.shape==data.shape:
            weighted_alpha = True
            alpha_img = get_img_from_array(alpha, brain_mask_file, image_type='SITK')
            alpha,bbox_physical_dimensions = gen_slices_array(alpha_img,selected_slices,slices_per_row,threshold=None, plane=plane)
        else:
            raise ValueError("alpha array must match the shape of the data array shape.")
    elif not isinstance(alpha, float) and not isinstance(alpha, int):
        raise ValueError("alpha must be a float of numpy array matching the data array shape.")

    # apply the thresholding to the data before plotting
    if percent_thresholding:
        threshold = threshold_top_percent(data, top_percent=top_percent)
        mask=np.abs(data)>=threshold # taking absolute values to include negative weights
        threshold_mask_img = get_img_from_array(mask.astype(float), brain_mask_file, image_type='SITK')
        sitk_img = sitk_img*threshold_mask_img

    # generate the array of slices to display and bbox dimensions to generate the figure
    data_slices,bbox_physical_dimensions = gen_slices_array(sitk_img,selected_slices,slices_per_row,threshold=fixed_threshold, plane=plane)
    
    if not fig and not ax: # automatically generate the plot if none is provided
        figsize=(bbox_physical_dimensions[0]*plot_size_scale,bbox_physical_dimensions[1]*plot_size_scale)
        print(f'Generating a figure with dimensions {figsize}')
        fig,ax = plt.subplots(1,1, figsize=figsize, constrained_layout=True)
    elif not ax or not fig:
        raise ValueError("""
                         Either both fig and ax are set to None, in which case the figure is 
                         automatically generated, or each parameter must be filled with an input.
                         """)

    # plot the template first as background if provided
    if template_file is not None:
        # resample the template to match the image
        scaled = otsu_scaling(template_file)
        scaled = sitk.Resample(scaled, sitk_img)
        slices_background,bbox_physical_dimensions = gen_slices_array(scaled,selected_slices,slices_per_row,threshold=None, plane=plane)
        pos = ax.imshow(slices_background, extent=[0,bbox_physical_dimensions[0],0,bbox_physical_dimensions[1]], vmin=0,vmax=1,cmap='gray', alpha=1, interpolation='none')
        ax.axis('off')

    pos = ax.imshow(data_slices, extent=[0,bbox_physical_dimensions[0],0,bbox_physical_dimensions[1]], vmin=vmin, vmax=vmax,cmap=cmap, alpha=alpha, interpolation='none')
    ax.axis('off')

    if plot_cbar:
        if weighted_alpha:
            # Create a separate mappable for the colorbar,
            # since the image itself has an array-valued alpha.
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap)

            cbar = fig.colorbar(sm, ax=ax)
        else:
            cbar = fig.colorbar(pos, ax=ax)
    else:
        cbar = None

    return fig,ax,cbar


def plot_many_brains(
        data_list,brain_mask_file,template_file=None,
        fig=None,ax_list=None,plot_size_scale=0.1,num_brain_per_row=1,
        plane='coronal', slice_selection=6, slices_per_row=None, # slice selection
        cmap='cold_hot', vmax=None, vmin=None, alpha=1, plot_cbar=False, # cmap range
        fixed_threshold=None,percent_thresholding=False,top_percent=10, # image thresholding
        ):
    
    """Plot a series of brain maps in one figure, with multiple rows and columns.

    Each brain map is plotted with :func:`plot_brain`, and this function takes all the same
    parameters except for ``data``, ``ax`` and ``fig``, which are replaced by the parameters below.

    Parameters
    ----------
    data_list : list of numpy.ndarray
        List of 1D arrays of shape (n_voxels,), each representing a brain map to plot.

    fig : matplotlib.figure.Figure, default=None
        A matplotlib figure. If None, the figure is automatically generated along with
        ``ax_list``.

    ax_list : list of matplotlib.axes.Axes, default=None
        List of axes of ``fig`` for plotting each entry of ``data_list``. If None, ``fig`` must
        also be None.

    num_brain_per_row : int, default=1
        If ``ax_list=None``, the number of brain maps to plot along a single row of the figure.
        Entries of ``data_list`` are plotted to fill a row from left to right before continuing
        onto the next row.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots.

    ax_list : list of matplotlib.axes.Axes
        The axes containing each plot.

    cbar_list : list
        The colorbar of each plot, as returned by :func:`plot_brain`.
    """

    selected_slices = select_slices_for_plotting(slice_selection, brain_mask_file, plane)
    print(f'Plotting along slices: {selected_slices}')

    num_plots = len(data_list)
    if not fig and not ax_list: # automatically generate the plot if none is provided

        # estimate the bounding box dimension for a single plot using the brain mask
        mask_img = sitk.ReadImage(brain_mask_file)
        slices,bbox_physical_dimensions = gen_slices_array(mask_img,selected_slices,slices_per_row,threshold=None, plane=plane)

        import math
        num_rows = math.ceil(num_plots/num_brain_per_row)

        figsize=(num_brain_per_row*bbox_physical_dimensions[0]*plot_size_scale,num_rows*bbox_physical_dimensions[1]*plot_size_scale)
        print(f'Generating a figure with dimensions {figsize}, {num_rows} rows and {num_brain_per_row} columns.')
        fig,axes = plt.subplots(num_rows,num_brain_per_row, figsize=figsize, constrained_layout=True)
        ax_list = axes.flatten()
    elif not ax_list or not fig:
        raise ValueError("""
                         Either both fig and ax are set to None, in which case the figure is 
                         automatically generated, or each parameter must be filled with an input.
                         """)
    
    if len(ax_list)<num_plots:
        raise ValueError("""
                         The number of brain maps in data_list is greater than the number of axes
                         in ax_list.
                         """)

    cbar_list = []
    for data,ax in zip(data_list,ax_list):
        fig,ax,cbar = plot_brain(data,brain_mask_file,template_file,
                    fig=fig,ax=ax,plot_size_scale=plot_size_scale,
                    plane=plane, slice_selection=selected_slices, slices_per_row=slices_per_row,
                    cmap=cmap, vmax=vmax, vmin=vmin, alpha=alpha, plot_cbar=plot_cbar,
                    fixed_threshold=fixed_threshold,percent_thresholding=percent_thresholding,top_percent=top_percent,
                    )
        cbar_list.append(cbar)

    return fig,ax_list,cbar_list


def select_slices_for_plotting(slice_selection, brain_mask_file, plane):
    """Select the slice indices to plot along one of the three cardinal planes.

    Parameters
    ----------
    slice_selection : list of int, numpy.ndarray, int or str
        If a list or 1D array of integers, these slice indices are used as is. If an int,
        that number of slices is selected at evenly spaced intervals across the slices
        containing brain voxels. If ``'all'``, all slices containing brain voxels are selected.

    brain_mask_file : str or pathlib.Path
        Full path to the brain mask Nifti file.

    plane : str
        The image plane among ``'sagittal'``, ``'coronal'``, or ``'horizontal'``.

    Returns
    -------
    selected_slices : list of int
        The array indices of the selected slices along ``plane``.
    """

    if isinstance(slice_selection, (list, tuple, np.ndarray)):
        selected_slices = np.asarray(slice_selection)
        if not selected_slices.ndim==1:
            raise ValueError(f"slice_selection is not a 1D array/list, input is of shape {selected_slices.shape}")
        if not np.issubdtype(selected_slices.dtype, np.integer):
            raise ValueError(f"slice_selection must be filled with integer values when provided as a list/array. Input is {slice_selection}.")
        if (selected_slices<0).any():
            raise ValueError(f"slice_selection must not contain negative indices. Input is {slice_selection}.")
        return selected_slices.tolist()
    else:
        def get_arr_slice(mask_arr, slicing_ax, idx): # takes out the 2D slice along a given axis/slice index value
            slicing = [slice(None)]*3
            slicing[slicing_ax] = idx
            return mask_arr[tuple(slicing)]

        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(brain_mask_file))
        # axes are in numpy (z,y,x) order, matching the indexing in gen_slices_array
        if plane=='sagittal':
            slicing_ax = 2
        elif plane=='coronal':
            slicing_ax = 1
        elif plane=='horizontal':
            slicing_ax = 0
        else:
            raise ValueError(f"{plane} is not among 'coronal', 'sagittal' or 'horizontal'.")

        num_slices = mask_arr.shape[slicing_ax]

        # create a list of slices that are within the brain mask - all other slices are discarded
        selected_slices = []
        for idx in range(num_slices):
            if get_arr_slice(mask_arr, slicing_ax, idx).any(): # only append index if there is at least 1 masked voxel
                    selected_slices.append(idx)
        # flip order to plot from front to back, or top to bottom
        selected_slices = selected_slices[::-1]

        if isinstance(slice_selection, str) and slice_selection=='all':
            return selected_slices
        elif isinstance(slice_selection, (int, np.integer)) and not isinstance(slice_selection, bool):
            if slice_selection<1:
                raise ValueError(f"slice_selection must be a positive integer when provided as an int. Got {slice_selection}.")
            # split the list of slices into 'slice_selection' # of bins, take the slice closest to the middle for each bins
            if slice_selection>len(selected_slices):
                print(f'{slice_selection} for slice_selection is greater than the number of available slices. All slices will be plotted.')
                return selected_slices
            indices = np.floor((np.arange(slice_selection) + 0.5) * len(selected_slices) / slice_selection).astype(int)
            selected_slices = [selected_slices[i] for i in indices]
            return selected_slices
        else:
            raise ValueError(f"slice_selection is not a list of integers, an integer, or 'all'. Got instead: {slice_selection}")


def gen_slices_array(sitk_img,selected_slices,slices_per_row=None,threshold=None, plane='coronal'):
    # this function assumes that selected_slices already provide the exact array indices along the desired plane

    physical_dimensions = (np.array(sitk_img.GetSpacing())*np.array(sitk_img.GetSize()))[::-1] # invert because the array is inverted indices
    array=sitk.GetArrayFromImage(sitk_img).astype(float)
    array[array==0]=None # set 0 values to be empty

    num_slices = len(selected_slices)

    if not threshold is None:
        array[np.abs(array)<threshold]=None

    if not slices_per_row:
        slices_per_row=num_slices # all slices in 1 row
    else:
        if num_slices<slices_per_row:
            raise ValueError(f"slices_per_row (={slices_per_row}) must be equal or smaller than num_slices (={num_slices})")


    if 'slices' in locals():
        raise # this variable should not exist a priori
    if plane=='sagittal':
        empty_slice = np.array([np.nan]).repeat(array.shape[0])[:,np.newaxis]
    elif plane=='coronal':
        empty_slice = np.array([np.nan]).repeat(array.shape[0])[:,np.newaxis]
    elif plane=='horizontal':
        empty_slice = np.array([np.nan]).repeat(array.shape[1])[:,np.newaxis]
    slices_row=empty_slice

    n=0
    i=0
    num_rows=0
    for s in selected_slices:
        # add a slice to the row
        if plane=='sagittal':
            slice=array[::-1,:,s].copy()
        elif plane=='coronal':
            slice=array[::-1,s,:].copy()
        elif plane=='horizontal':
            slice=array[s,::-1,:].copy()

        slices_row=np.concatenate((slices_row,slice,empty_slice),axis=1)
        n+=1
        i+=1
        if n==slices_per_row or i==num_slices: # append the row if the max per row or the last slice is reached
            if 'slices' in locals(): # a previous row exists
                if i==num_slices:
                    # the last row may not be filled with slices - fill the rest with NaN values
                    last_row = np.full((slices_row.shape[0],slices.shape[1]), np.nan)
                    last_row[:,:slices_row.shape[1]] = slices_row
                    slices_row = last_row
                slices=np.concatenate((slices,slices_row),axis=0) # concatenate the completed row
            else: # this is the first row
                slices=slices_row
            num_rows+=1
            slices_row=empty_slice # start a new row
            n=0 # restart the count

    if plane=='sagittal':
        bbox_physical_dimensions = [physical_dimensions[1]*slices_per_row,physical_dimensions[0]*num_rows]
    elif plane=='coronal':
        bbox_physical_dimensions = [physical_dimensions[2]*slices_per_row,physical_dimensions[0]*num_rows]
    elif plane=='horizontal':
        bbox_physical_dimensions = [physical_dimensions[2]*slices_per_row,physical_dimensions[1]*num_rows]

    return slices,bbox_physical_dimensions
