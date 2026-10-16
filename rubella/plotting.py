import matplotlib.pyplot as plt
import nilearn.plotting # this is imported so that 'cold_hot' colormap is available
from rabies.visualization import otsu_scaling
from rabies.analysis_pkg.diagnosis_pkg.dataset_QC import threshold_top_percent
from rabies.utils import recover_3D
import SimpleITK as sitk
import numpy as np

def plot_brain(data,brain_mask_file,template_file=None,
               fig=None,ax=None,plot_size_scale=0.1,
               plane='coronal', selected_slices='all', slices_per_row=None, # slice selection
               cmap='cold_hot', vmax=None, vmin=None, alpha=1, plot_cbar=False, # cmap range
               fixed_threshold=None,percent_thresholding=False,top_percent=10, # image thresholding
               ):
    '''
    Function for plotting a single brain map as a series of brain slices along one of the three cardinal plane,
    overlayed on top of a template in the same space.

    Parameters
    ----------
    data : 1D array
        The vector representing the brain map. The vector indices must have been previously generated using brain_mask_file.

    brain_mask_file : str
        Full path to the brain mask Nifti file 

    template_file : str
        File path to the background image

    fig : matplotlib.pyplot.figure
        A matplotlib figure. If None, then the figure is automatically generated
        based on the finalized brain map plot dimensions.

    ax : matplotlib.pyplot.axis
        The ax associated to fig. If None, there must also be fig=None
    
    scale : float
        A scaling factor for controlling the figure size when fig=None.

    plane : str
        Select a image plane among 'sagittal', 'coronal', or 'horizontal'

    selected_slices : list or 'all'
        The list of slice indices along the desired plane to plot. If 'all'
        is inputed, all slices are plotted.
    
    slices_per_row : int
        This controls the number of slices to plot in one row within the plot.
        If None, all slices are plotted in one row.
        
    cmap : str
        Input to the matplotlib colormap.

    vmax : float
        The max value for the colormap. If None, it is set to the value of the 98th percentile
        of the voxel value distribution.

    vmin : float
        The minimum value for the colormap. If None, it is set to -vmax.

    alpha : float, np.ndarray
        The transparency level of the brain map overlay. Must be between 0 and 1.
        It can be an array of the same shape as data to provide a per voxel alpha
        value.

    plot_cbar : bool
        Whether to plot a colorbar.
                       
    fixed_threshold : float
        Every value below this threshold are hidden in the plot.

    percent_thresholding : bool
        Whether to automatically threshold the image to only capture the top X% 
        of voxels with highest value. The percentage is set by top_percent.
        
    top_percent : float
        The percentage value for percent_thresholding.
    '''

    weighted_alpha = False
    if isinstance(alpha, np.ndarray):
        if alpha.shape==data.shape:
            alpha_img = recover_3D(brain_mask_file, alpha)
            weighted_alpha = True
        else:
            raise ValueError("alpha array must match the shape of the data array shape.")
    elif not isinstance(alpha, float) and not isinstance(alpha, int):
        raise ValueError("alpha must be a float of numpy array matching the data array shape.")
    sitk_img = recover_3D(brain_mask_file, data)

    if selected_slices=='all':
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(brain_mask_file))
        selected_slices = []
        if plane=='sagittal':
            for i in range(mask_arr.shape[2]):
                if mask_arr[:,:,i].max()==1: # only append index if there is at least 1 masked voxel
                    selected_slices.append(i)
        elif plane=='coronal':
            for i in range(mask_arr.shape[1]):
                if mask_arr[:,i,:].max()==1: # only append index if there is at least 1 masked voxel
                    selected_slices.append(i)
        elif plane=='horizontal':
            for i in range(mask_arr.shape[0]):
                if mask_arr[i,:,:].max()==1: # only append index if there is at least 1 masked voxel
                    selected_slices.append(i)
        print(f'Plotting across {len(selected_slices)} slices')
        # flip order to plot from front to back, or top to bottom
        selected_slices = selected_slices[::-1]

    if percent_thresholding:
        threshold = threshold_top_percent(data, top_percent=top_percent)
        mask=np.abs(data)>=threshold # taking absolute values to include negative weights
        threshold_mask_img = recover_3D(brain_mask_file,mask)
        sitk_img = sitk_img*threshold_mask_img
        
    if template_file is not None:
        # resample the template to match the image
        scaled = otsu_scaling(template_file)
        scaled = sitk.Resample(scaled, sitk_img)
        slices_background,bbox_physical_dimensions = gen_slices_array(scaled,selected_slices,slices_per_row,threshold=None, plane=plane)

    # select vmax at 98th percentile value
    vector = data.flatten()
    vector.sort()
    if vmax is None:
        vmax = vector[int(len(vector)*0.98)]
    if vmin is None:
        vmin=-vmax

    slices,bbox_physical_dimensions = gen_slices_array(sitk_img,selected_slices,slices_per_row,threshold=fixed_threshold, plane=plane)
    
    if not fig and not ax: # automatically generate the plot if none is provided
        figsize=(bbox_physical_dimensions[0]*plot_size_scale,bbox_physical_dimensions[1]*plot_size_scale)
        print(f'Generating a figure with dimensions {figsize}')
        fig,ax = plt.subplots(1,1, figsize=figsize, constrained_layout=True)
    elif not ax or not fig:
        raise ValueError("""
                         Either both fig and ax are set to None, in which case the figure is 
                         automatically generated, or each parameter must be filled with an input.
                         """)

    if template_file is not None:
        pos = ax.imshow(slices_background, extent=[0,bbox_physical_dimensions[0],0,bbox_physical_dimensions[1]], vmin=0,vmax=1,cmap='gray', alpha=1, interpolation='none')
        ax.axis('off')

    if weighted_alpha:
        alpha,bbox_physical_dimensions = gen_slices_array(alpha_img,selected_slices,slices_per_row,threshold=None, plane=plane)
            
    pos = ax.imshow(slices, extent=[0,bbox_physical_dimensions[0],0,bbox_physical_dimensions[1]], vmin=vmin, vmax=vmax,cmap=cmap, alpha=alpha, interpolation='none')
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
        plane='coronal', selected_slices='all', slices_per_row=None, # slice selection
        cmap='cold_hot', vmax=None, vmin=None, alpha=1, plot_cbar=False, # cmap range
        fixed_threshold=None,percent_thresholding=False,top_percent=10, # image thresholding
        ):
    
    '''
    Function for plotting a series of brain in one figure with multiple rows/columns.

    Parameters
    ----------
    This function inherits all parameters from plot_brain(), except the following:

    data_list : list
        List of 1D vector brain maps to plot.

    ax_list : list
        List of matplotlib axes for plotting data_list. If None, the figure is automatically generated along 
        with the list of axes.

    num_brain_per_row : int
        If ax_list=None, this controls the number of brain maps to plot along a single row of the figure.
        Elements in data_list are plotted to fill a row from left to right before continuing onto the next row.

    '''
    
    if selected_slices=='all':
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(brain_mask_file))
        selected_slices = []
        if plane=='sagittal':
            for i in range(mask_arr.shape[2]):
                if mask_arr[:,:,i].max()==1: # only append index if there is at least 1 masked voxel
                    selected_slices.append(i)
        elif plane=='coronal':
            for i in range(mask_arr.shape[1]):
                if mask_arr[:,i,:].max()==1: # only append index if there is at least 1 masked voxel
                    selected_slices.append(i)
        elif plane=='horizontal':
            for i in range(mask_arr.shape[0]):
                if mask_arr[i,:,:].max()==1: # only append index if there is at least 1 masked voxel
                    selected_slices.append(i)
        print(f'Plotting across {len(selected_slices)} slices')
        # flip order to plot from front to back, or top to bottom
        selected_slices = selected_slices[::-1]

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
                    plane=plane, selected_slices=selected_slices, slices_per_row=slices_per_row,
                    cmap=cmap, vmax=vmax, vmin=vmin, alpha=alpha, plot_cbar=plot_cbar,
                    fixed_threshold=fixed_threshold,percent_thresholding=percent_thresholding,top_percent=top_percent,
                    )
        cbar_list.append(cbar)

    return fig,ax_list,cbar_list


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
