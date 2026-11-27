"""Seed-based connectivity maps and voxelwise one-sample statistics."""

from pathlib import Path
from time import monotonic
import warnings

import SimpleITK as sitk
import colorcet as cc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from rabies.utils import recover_3D

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from scipy.signal import savgol_filter

from scipy.stats import t as student_t
from scipy.stats import norm, ttest_1samp, wilcoxon, rankdata, mannwhitneyu, ttest_ind

from statsmodels.stats.multitest import multipletests
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from rubella.basic_fc import seed_based_FC

###################################################################################################################################################################

############################################################################# DISPLAY #############################################################################

###################################################################################################################################################################


def plot_img(
    data,
    mask_path,
    template_path=None,
    seed_path=None,
    slice_ids=np.linspace(5, 25, 15).astype(int),
    nb_row=3,
    vmin=-0.3,
    vmax=0.3,
    th=None,
    cmap=cc.cm["CET_D1"],
    fig=None,
    ax=None,
    box_width=5,
    alpha=0.7,
):
    """
    Plot a masked voxelwise map as a montage of brain slices.

    Reconstruct the map in the brain mask's voxel grid and display slices
    along the second axis of the SimpleITK array. Optionally overlay the
    map on an anatomical template and draw the seed ROI boundary.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional array of values for the voxels inside the brain
        mask, in the voxel order expected by ``recover_3D``.
    mask_path : str or pathlib.Path
        Path to the brain mask that defines the voxel grid and indexing.
    template_path : str or pathlib.Path, optional
        Path to an anatomical image on the brain mask's voxel grid.
        If None, use a black background.
    seed_path : str or pathlib.Path, optional
        Path to a binary seed ROI on the brain mask's voxel grid.
        If provided, draw its boundary as a black contour.
    slice_ids : array-like of int, optional
        Zero-based indices along the second axis of the reconstructed
        array. By default, use 15 evenly spaced values from 5 through 25,
        converted to integers.
    nb_row : int, default=3
        Number of columns in a newly created subplot grid.
    vmin : float, default=-0.3
        Lower color limit for the map overlay.
    vmax : float, default=0.3
        Upper color limit for the map overlay.
    th : float or numpy.ndarray, optional
        A numeric scalar displays voxels whose absolute map value is
        strictly greater than the threshold. A one-dimensional array
        supplies opacity weights in [0, 1], in the same voxel order as
        ``data``. If None, use the brain mask.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the map overlay. Defaults to Colorcet's ``CET_D1``.
    fig : matplotlib.figure.Figure, optional
        Existing figure. Supply together with ``ax``.
    ax : matplotlib.axes.Axes or array-like of matplotlib.axes.Axes, optional
        Axes belonging to ``fig``, used in row-major order. Supply at
        least one axis per slice.
    box_width : float, default=5
        Width of a newly created figure, in inches.
    alpha : float, default=0.7
        Overlay opacity in [0, 1], multiplied by the brain mask, the
        scalar threshold result, or the voxelwise opacity weights.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the slice montage.
    ax : matplotlib.axes.Axes or array-like of matplotlib.axes.Axes
        The supplied axes, or a two-dimensional array of newly created
        axes.

    Raises
    ------
    ValueError
        If only one of ``fig`` and ``ax`` is supplied or there are fewer
        supplied axes than requested slices.

    Notes
    -----
    Each slice is rotated by 180 degrees. Template intensities are
    squared before display.
    """
    
    if (fig is None) != (ax is None):
        raise ValueError("Supply fig and ax together.")

    nb_slices = len(slice_ids)
    if ax is not None and np.size(ax) < nb_slices:
        raise ValueError("Supply at least one axis per slice.")

    # Loads image data and parameters

    mask_path = str(mask_path)
    img = sitk.GetArrayFromImage(recover_3D(mask_path, data)).astype(float)
    mask_img = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(bool)

    # Checks input and raises errors
    
    template_img = None
    if template_path is not None:
        template_img = sitk.GetArrayFromImage(
            sitk.ReadImage(str(template_path))
        ).astype(float)

    seed_img = None
    if seed_path is not None:
        seed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(seed_path)))

    scalar_threshold = th is not None and np.isscalar(th)
    opacity_img = mask_img
    if th is not None and not scalar_threshold:
        opacity_img = sitk.GetArrayFromImage(
            recover_3D(mask_path, np.asarray(th))
        ).astype(float)

    # initiates figure if needed
    
    if fig is None:
        nb_lines = int(np.ceil(nb_slices / nb_row))
        box_height = box_width * img.shape[0] / img.shape[1]
        fig, ax = plt.subplots(
            nb_lines,
            nb_row,
            squeeze=False,
            figsize=(box_width, box_height),
            dpi=200,
            facecolor="black",
            gridspec_kw={"wspace": 0, "hspace": 0},
        )

    axes = np.atleast_1d(ax).ravel()
    for axis in axes:
        axis.axis("off")
        axis.set_facecolor("black")

    # displays ax-wise
        
    for axis, slice_id in zip(axes, slice_ids):
        if template_img is not None:
            template_slice = np.rot90(template_img[:, slice_id, :] ** 2, 2)
            axis.imshow(template_slice, cmap="Greys_r")

        map_slice = np.rot90(img[:, slice_id, :], 2)
        if scalar_threshold:
            opacity = np.abs(map_slice) > th
        else:
            opacity = np.rot90(opacity_img[:, slice_id, :], 2)

        axis.imshow(
            map_slice,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha * opacity,
        )

        if seed_img is not None:
            seed_slice = np.rot90(seed_img[:, slice_id, :], 2)
            axis.contour(
                seed_slice, levels=[0.5], colors=["black"], linewidths=[1.5]
            )

    # tighten display
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()
         
    return fig, ax

def plot_add_contour(
    fig,
    ax,
    data,
    mask_path,
    slice_ids=None,
    levels=(0.5,),
    colors=("w",),
    linestyles=(":",),
    linewidths=(0.5,),
):
    """
    Overlay contours on an existing brain slice montage.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the montage.
    ax : matplotlib.axes.Axes or array-like of matplotlib.axes.Axes
        Montage axes in row-major order, as returned by ``plot_img``.
    data : numpy.ndarray
        One-dimensional array of values inside the brain mask, in the
        voxel order expected by ``recover_3D``.
    mask_path : str or pathlib.Path
        Path to the brain mask used to create the montage.
    slice_ids : array-like of int, optional
        Slice indices in the same order as in the original montage.
        If None, use the default slice indices from ``plot_img``.
    levels : array-like of float, default=(0.5,)
        Contour levels in increasing order. Use 0.5 for a binary 0/1 map.
    colors : color or sequence of colors, default=("w",)
        Contour colors.
    linestyles : str or sequence of str, default=(":",)
        Contour line styles.
    linewidths : float or sequence of float, default=(0.5,)
        Contour line widths in points.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The input figure with contours added.
    ax : matplotlib.axes.Axes or array-like of matplotlib.axes.Axes
        The input axes.

    Raises
    ------
    ValueError
        If there are fewer axes than requested slices.
    """

    # input checks and raises errors
    
    if slice_ids is None:
        slice_ids = np.linspace(5, 25, 15).astype(int)

    axes = np.atleast_1d(ax).ravel()
    if axes.size < len(slice_ids):
        raise ValueError("Supply at least one axis per slice.")

    # from vect to image
    
    img = sitk.GetArrayFromImage(recover_3D(str(mask_path), data))

    # displays axis-wise
    
    for axis, slice_id in zip(axes, slice_ids):
        contour_slice = np.rot90(img[:, slice_id, :], 2)
        axis.contour(
            contour_slice,
            levels=levels,
            colors=colors,
            linestyles=linestyles,
            linewidths=linewidths,
        )

    return fig, ax

def plot_cum_CI(
    vals,
    sd,
    cip,
    cim,
    fig = None,
    ax = None,
    labels = None
):

    """Plot ranked estimates with confidence intervals and optional SD curves.

    Parameters
    ----------
    vals : array-like of float
        One-dimensional array of at least two finite estimates. The
        estimates are plotted in ascending order.
    sd : array-like of float or None
        Nonnegative standard deviations, one per estimate. If provided,
        draw dotted curves at the estimate plus and minus the smoothed
        SD. Smoothing is applied after sorting by estimate.
    cip : array-like of float
        Upper confidence bounds in the original order of `vals`.
        Supply absolute bounds in the same units as the estimates.
    cim : array-like of float
        Lower confidence bounds in the original order of `vals`.
        Supply absolute bounds in the same units as the estimates.
    fig : matplotlib.figure.Figure or None, optional
        Figure to draw into. If `ax` is provided, `fig` must be the figure
        that owns it. If both are None, create a new figure.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into. If None, create axes in `fig`.
    labels : sequence of str or None, optional
        Legend labels in the order estimate, confidence interval, and
        optionally SD. Supply two labels if `sd` is None and three
        otherwise. If None, omit the legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    new_order : numpy.ndarray
        Indices that sort `vals` in ascending order. Equal estimates
        retain their original order.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes or nonfinite values,
        a lower bound exceeds its upper bound, an SD is negative,
        the number of labels is incorrect, or `fig` and `ax` disagree.

    Notes
    -----
    The shaded area connects the supplied intervals in rank order.
    The function performs no confidence interval calculation.
    """
    
    vals, cip, cim = (np.asarray(a, dtype=float) for a in (vals, cip, cim))
    if vals.ndim != 1 or vals.size < 2:
        raise ValueError("vals must be one-dimensional with at least two values.")

    if cip.shape != vals.shape or cim.shape != vals.shape:
        raise ValueError("cip and cim must have the same shape as vals.")
    if not all(np.isfinite(a).all() for a in (vals, cip, cim)):
        raise ValueError("vals, cip, and cim must contain only finite values.")
    if np.any(cim > cip):
        raise ValueError("Each lower confidence bound must be <= its upper bound.")

    if sd is not None:
        sd = np.asarray(sd, dtype=float)
        if sd.shape != vals.shape or not np.isfinite(sd).all() or np.any(sd < 0):
            raise ValueError("sd must match vals and contain finite, nonnegative values.")

    n_labels = 2 if sd is None else 3
    if labels is None:
        labels = (None,) * n_labels
    elif isinstance(labels, str) or len(labels) != n_labels:
        raise ValueError(f"Supply {n_labels} legend labels.")

    if ax is not None:
        if fig is not None and ax.figure is not fig:
            raise ValueError("ax must belong to fig.")
        fig = ax.figure
    elif fig is None:
        fig, ax = plt.subplots(figsize=(15, 5), dpi=200, facecolor="black")
    else:
        ax = fig.add_subplot()

    ax.set_facecolor("black")
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("white")

    new_order = np.argsort(vals, kind="stable")
    ranks = np.arange(vals.size)
    estimates = vals[new_order]
    lower, upper = cim[new_order], cip[new_order]

    ax.plot(ranks, estimates, color="C1", label=labels[0])
    ax.fill_between(ranks, lower, upper, color="C1", alpha=0.2, label=labels[1])

    ymin = min(0.0, lower.min(), estimates.min())
    ymax = max(0.0, upper.max(), estimates.max())

    if sd is not None:
        spread = sd[new_order]
        window = min(99, vals.size)
        if window % 2 == 0:
            window -= 1
        if window >= 5:
            spread = np.maximum(savgol_filter(spread, window, 3), 0.0)

        sd_lower, sd_upper = estimates - spread, estimates + spread
        ax.plot(ranks, sd_upper, color="C1", ls=":", lw=0.5, label=labels[2])
        ax.plot(ranks, sd_lower, color="C1", ls=":", lw=0.5)
        ymin = min(ymin, sd_lower.min())
        ymax = max(ymax, sd_upper.max())

    padding = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
    ax.axhline(0, color="white", ls=":", lw=0.5, zorder=0)
    ax.set_xlim(-0.5, vals.size - 0.5)
    ax.set_xticks([])
    ax.set_yticks(np.unique([0.0, estimates[-1]]))
    ax.set_ylim(ymin - padding, ymax + padding)
    ax.tick_params(axis="y", colors="white")

    if any(label is not None for label in labels):
        ax.legend(facecolor="black", edgecolor="white", labelcolor="white", framealpha=1)

    return fig, ax, new_order
from scipy.stats import mannwhitneyu, ttest_ind

###################################################################################################################################################################

############################################################################## STATS ##############################################################################

###################################################################################################################################################################

def perform_SBMs(
    data_obj,
    seed_df,
    display_fig=True,
    template_path=None,
    save_dir_png=None,
    save_output=True,
    save_dir_nii=None,
    n_jobs=4,
    vmin = -0.3,
    vmax = 0.3
):
    
    """
    Compute individual Fisher-z seed-based functional connectivity maps.
    
    For each seed, correlate its mean time series with every voxel time
    series within the brain mask, then apply the Fisher transformation
    ``z = arctanh(r)``. Optionally save maps as compressed NIfTI files and
    generate PNG figures.
    
    Parameters
    ----------
    data_obj : rubella.data_obj.DataObject
        Loaded data object providing ``timeseries_dict``, ``brain_mask_file``,
        and ``img_id_list``. Each time-series array must have shape
        (n_timepoints, n_mask_voxels).
    seed_df : pandas.DataFrame
        Seed definitions with columns ``"acro"`` (unique seed acronyms) and
        ``"path"`` (paths to seed ROI images compatible with the brain mask).
    display_fig : bool, default=True
        Generate and save a PNG figure for each seed and image ID.
        If True, save a PNG figure for each seed and image ID, then close the figure.
    template_path : str or pathlib.Path, optional
        Anatomical background image used for plotting. Required when
        ``display_fig=True``.
    save_dir_png : str or pathlib.Path, optional
        Output directory for PNG figures. Required when ``display_fig=True``.
        Created, including missing parent directories, if necessary.
    save_output : bool, default=True
        Save each connectivity map as a compressed float32 NIfTI image,
        using the spatial metadata of the brain mask.
    save_dir_nii : str or pathlib.Path, optional
        Output directory for NIfTI maps. Required when ``save_output=True``.
        Created, including missing parent directories, if necessary.
    n_jobs : int, default=4
        Number of concurrent jobs, with one seed processed per job.
        Use 1 for sequential execution or -1 for all available CPUs.
        Thread-based execution is preferred.
    vmin : float, default=-0.3
        Lower color limit passed to ``plot_img``.
    vmax : float, default=0.3
        Upper color limit passed to ``plot_img``.
    
    Returns
    -------
    SBMs : dict
        Dictionary mapping each seed acronym to a list of one-dimensional
        Fisher-z connectivity arrays, each of shape (n_mask_voxels,).
        Lists follow ``data_obj.img_id_list`` order; dictionary keys follow
        the seed DataFrame order. Returns an empty dictionary if no seeds
        are provided and output options pass validation.
    
    Raises
    ------
    ValueError
        If seed acronyms are duplicated, plotting is enabled without
        ``template_path`` or ``save_dir_png``, or NIfTI saving is enabled
        without ``save_dir_nii``.
    
    Notes
    -----
    Computation and NIfTI saving run in parallel across seeds. Plotting runs
    sequentially in the main thread. A notebook-oriented Rich progress bar
    advances after each seed's requested outputs have been processed.
    
    Output filenames follow ``ROI-{acro}_{img_id}_z.png`` and
    ``ROI-{acro}_{img_id}_z.nii.gz``. Existing files with matching names
    are overwritten.
    
    Fisher-z values are transformed correlation coefficients, not
    standardized test statistics.
    """

    mask_path = str(data_obj.brain_mask_file)
    img_ids = list(data_obj.img_id_list)
    seeds = list(zip(seed_df["acro"], seed_df["path"]))

    seed_acros = [acro for acro, _ in seeds]

    if len(set(seed_acros)) != len(seed_acros):
        raise ValueError("Seed acronyms must be unique.")

    # Validate options before starting computation.
    if display_fig:
        if template_path is None:
            raise ValueError("display_fig=True requires template_path.")

        if save_dir_png is None:
            raise ValueError("display_fig=True requires save_dir_png.")

        save_dir_png = Path(save_dir_png)
        save_dir_png.mkdir(parents=True, exist_ok=True)

        import nilearn.plotting  # noqa: F401 - Registers the cold_hot colormap.

    if save_output:
        if save_dir_nii is None:
            raise ValueError("save_output=True requires save_dir_nii.")

        save_dir_nii = Path(save_dir_nii)
        save_dir_nii.mkdir(parents=True, exist_ok=True)

    if not seeds:
        return {}

    def process_seed(seed_acro, seed_path):
        """Compute and optionally save all subject maps for one seed."""

        results = seed_based_FC(
            data_obj.timeseries_dict,
            mask_path,
            img_ids,
            str(seed_path),
            z_corr=True,
        )

        if save_output:
            for img_id in img_ids:
                recovered = recover_3D(mask_path, results[img_id])
                recovered = sitk.Cast(recovered, sitk.sitkFloat32)

                filename = f"ROI-{seed_acro}_{img_id}_z.nii.gz"

                sitk.WriteImage(
                    recovered,
                    str(save_dir_nii / filename),
                    useCompression=True,
                )

        return seed_acro, seed_path, results

    SBMs = {}

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("Elapsed"),
        TimeElapsedColumn(),
        TextColumn("Remaining"),
        TimeRemainingColumn(),
        console=Console(force_jupyter=True),
        auto_refresh=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:

        task = progress.add_task(
            "Seed-based maps",
            total=len(seeds),
        )
        progress.refresh()
        last_refresh = monotonic()

        with Parallel(
            n_jobs=n_jobs,
            prefer="threads",
            batch_size=1,
            pre_dispatch="2*n_jobs",
            return_as="generator_unordered",
        ) as parallel:

            completed_seeds = parallel(
                delayed(process_seed)(seed_acro, seed_path)
                for seed_acro, seed_path in seeds
            )

            for seed_acro, seed_path, results in completed_seeds:
                SBMs[seed_acro] = [
                    results[img_id] for img_id in img_ids
                ]

                # Keep Matplotlib operations in the main thread.
                if display_fig:
                    for img_id in img_ids:
                        fig, _ = plot_img(
                            results[img_id],
                            mask_path,
                            template_path=str(template_path),
                            seed_path=str(seed_path),
                            slice_ids=np.linspace(10, 25, 15).astype(int),
                            cmap="cold_hot",
                            vmin = vmin,
                            vmax = vmax
                        )

                        try:
                            fig.suptitle(
                                img_id.split("_ses")[0].replace("_", " "),
                                color="white",
                                fontsize=10,
                            )

                            filename = f"ROI-{seed_acro}_{img_id}_z.png"
                            fig.savefig(save_dir_png / filename)

                        finally:
                            plt.close(fig)

                # A seed is complete after its requested outputs are saved.
                progress.update(task, advance=1, refresh=False)

                now = monotonic()
                if now - last_refresh >= 0.5:
                    progress.refresh()
                    last_refresh = now

        progress.refresh()

    # Restore the input seed order after unordered completion.
    return {seed_acro: SBMs[seed_acro] for seed_acro in seed_acros}


def _run_jobs(worker, items, *, n_jobs, show_progress, description):
    """Run jobs and return results in input order."""
    if not show_progress or n_jobs == 1 or len(items) < 2:
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(worker)(item) for item in items
        )

    def run_item(index, item):
        return index, worker(item)

    results = [None] * len(items)
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("Elapsed"),
        TimeElapsedColumn(),
        TextColumn("Remaining"),
        TimeRemainingColumn(),
        console=Console(),
        auto_refresh=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        task = progress.add_task(description, total=len(items))
        progress.refresh()
        last_refresh = monotonic()

        with Parallel(
            n_jobs=n_jobs,
            prefer="threads",
            batch_size=1,
            return_as="generator_unordered",
        ) as parallel:
            completed = parallel(
                delayed(run_item)(index, item) for index, item in enumerate(items)
            )
            for index, result in completed:
                results[index] = result
                progress.advance(task)
                now = monotonic()
                if now - last_refresh >= 0.5:
                    progress.refresh()
                    last_refresh = now

        progress.refresh()
    return results

###################################################################################################################################################################

############################################################################ ONE SAMPLE ###########################################################################

###################################################################################################################################################################

def estimate_maps_one_sample(
    group_matrix,
    n_jobs=4,
    confidence_level=0.95,
    n_bootstrap=2000,
    chunk_size=128,
    batch_size=32,
    seed=0,
    *,
    show_progress=True,
):
    """
    Estimate voxelwise summaries and pointwise confidence intervals.

    Parameters
    ----------
    group_matrix : array-like of float
        Finite observations of shape (n_subjects, n_voxels), with at least
        two independent subjects and one voxel. Columns must use the
        same voxel order across subjects.
    n_jobs : int, default=4
        Number of joblib workers for the bootstrap. Use 1 for sequential
        execution or -1 for all available CPUs.
    confidence_level : float or None, default=0.95
        Confidence level between zero and one. Use None to return
        descriptive summaries without confidence intervals or bootstrapping.
    n_bootstrap : int, default=2000
        Number of subject-level bootstrap resamples. Must be at least two.
    chunk_size : int, default=128
        Number of voxels processed per parallel job.
    batch_size : int, default=32
        Number of bootstrap resamples evaluated together within each job.
        Smaller values reduce temporary memory use.
    seed : int or None, default=0
        Random seed. All voxel chunks use the same subject resamples.
    show_progress : bool, default=True
        Display a Rich progress bar for parallel bootstrap runs, with
        one step per completed voxel chunk.

    Returns
    -------
    result : dict
        ``result["mean"]`` contains ``estimate``, ``sd``, and ``se``.
        ``result["median"]`` contains ``estimate`` and ``mad``.
        ``result["cohens_d"]`` contains ``estimate``, the mean divided
        by the sample standard deviation. The MAD is unscaled.

        When confidence intervals are requested, each entry also contains
        ``ci`` of shape (n_voxels, 2), ordered as lower and upper bounds.
        Median and Cohen's d entries also contain bootstrap ``se``.
        The Cohen's d entry additionally contains
        ``bootstrap_valid_fraction``. All other arrays have shape (n_voxels,).

    Raises
    ------
    ValueError
        If the input shape, input values, or bootstrap settings are invalid.

    Notes
    -----
    Confidence intervals are pointwise. The mean uses a Student-t interval.
    The median and Cohen's d use percentile bootstrap intervals.

    Voxels with zero sample variance have NaN mean intervals and NaN
    Cohen's d estimates. If any resample has zero variance, the Cohen's d
    bootstrap interval and standard error for that voxel are NaN.
    """
    values = np.asarray(group_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Expected shape (n_subjects >= 2, n_voxels >= 1).")
    if not np.isfinite(values).all():
        raise ValueError("group_matrix must contain only finite values.")

    n_subjects, n_voxels = values.shape
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    mean_se = sd / np.sqrt(n_subjects)
    median = np.median(values, axis=0)
    cohens_d = np.divide(
        mean, sd, out=np.full(n_voxels, np.nan), where=sd > 0
    )

    result = {
        "mean": {"estimate": mean, "sd": sd, "se": mean_se},
        "median": {
            "estimate": median,
            "mad": np.median(np.abs(values - median), axis=0),
        },
        "cohens_d": {"estimate": cohens_d},
    }
    if confidence_level is None:
        return result

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")

    for name, value, minimum in (
        ("n_bootstrap", n_bootstrap, 2),
        ("chunk_size", chunk_size, 1),
        ("batch_size", batch_size, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    tail = (1 - confidence_level) / 2
    quantiles = (tail, 1 - tail)
    critical_t = student_t.ppf(1 - tail, df=n_subjects - 1)
    half_width = critical_t * mean_se
    mean_ci = np.column_stack((mean - half_width, mean + half_width))
    mean_ci[sd == 0] = np.nan
    result["mean"]["ci"] = mean_ci

    rng = np.random.default_rng(seed)
    subject_indices = rng.integers(
        n_subjects, size=(n_bootstrap, n_subjects)
    )

    def bootstrap_chunk(start):
        stop = min(start + chunk_size, n_voxels)
        chunk = values[:, start:stop]
        bootstrap_median = np.empty((n_bootstrap, stop - start))
        bootstrap_d = np.full((n_bootstrap, stop - start), np.nan)

        for first in range(0, n_bootstrap, batch_size):
            batch = slice(first, first + batch_size)
            samples = chunk[subject_indices[batch]]
            bootstrap_median[batch] = np.median(samples, axis=1)
            sample_sd = samples.std(axis=1, ddof=1)
            np.divide(
                samples.mean(axis=1),
                sample_sd,
                out=bootstrap_d[batch],
                where=sample_sd > 0,
            )

        return (
            bootstrap_median.std(axis=0, ddof=1),
            np.quantile(bootstrap_median, quantiles, axis=0).T,
            bootstrap_d.std(axis=0, ddof=1),
            np.quantile(bootstrap_d, quantiles, axis=0).T,
            np.isfinite(bootstrap_d).mean(axis=0),
        )

    chunks = _run_jobs(
        bootstrap_chunk,
        range(0, n_voxels, chunk_size),
        n_jobs=n_jobs,
        show_progress=show_progress,
        description="Bootstrap voxel chunks",
    )
    median_se, median_ci, d_se, d_ci, d_valid_fraction = (
        np.concatenate(parts, axis=0) for parts in zip(*chunks)
    )

    result["median"].update(se=median_se, ci=median_ci)
    result["cohens_d"].update(
        se=d_se, ci=d_ci, bootstrap_valid_fraction=d_valid_fraction
    )
    return result


def compute_stats_one_sample(
    group_matrix, tests=("ttest", "wilcoxon"), *, n_jobs=1, show_progress=True
):
    """
    Compute two-sided one-sample tests across subjects at each voxel.

    Parameters
    ----------
    group_matrix : array-like of float
        Array of shape (n_subjects, n_voxels), with at least two
        independent subjects and one voxel. Columns must follow the
        same voxel order across subjects.
    tests : str or sequence of {"ttest", "wilcoxon"}, optional
        Tests to compute. The t-test tests for a zero mean. The
        Wilcoxon signed-rank test tests for symmetry about zero.
    n_jobs : int, default=1
        Number of joblib workers. With both tests selected, they can
        run concurrently. Each test receives the full set of its
        eligible voxels in one SciPy call.
    show_progress : bool, default=True
        Display a Rich progress bar when running both tests in parallel,
        with one step per completed test.

    Returns
    -------
    results : dict
        Each test has ``statistic``, ``pvalues``, and ``valid_voxels``
        arrays of shape (n_voxels,). The t-statistic is signed.
        The two-sided Wilcoxon statistic is the smaller rank sum.

    Notes
    -----
    Voxels with nonfinite observations are excluded from both tests.
    Constant voxels are excluded from the t-test. All-zero voxels are
    excluded from Wilcoxon. Excluded voxels have NaN statistics and
    p-values.
    """
    values = np.asarray(group_matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Expected shape (n_subjects >= 2, n_voxels >= 1).")

    tests = (tests,) if isinstance(tests, str) else tuple(tests)
    if not tests or any(test not in ("ttest", "wilcoxon") for test in tests):
        raise ValueError("Select 'ttest', 'wilcoxon', or both.")

    finite = np.isfinite(values).all(axis=0)
    tests = tuple(dict.fromkeys(tests))
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
    ):
        raise ValueError("n_jobs must be a nonzero integer.")

    def run_test(test):
        statistic = np.full(values.shape[1], np.nan)
        pvalues = np.full(values.shape[1], np.nan)

        if test == "ttest":
            eligible = finite & np.any(values != values[0], axis=0)
        else:
            eligible = finite & np.any(values != 0, axis=0)

        if eligible.any():
            eligible_values = values[:, eligible]
            if test == "ttest":
                result = ttest_1samp(eligible_values, popmean=0, axis=0)
            else:
                result = wilcoxon(eligible_values, axis=0)
            statistic[eligible] = result.statistic
            pvalues[eligible] = result.pvalue

        entry = {
            "statistic": statistic,
            "pvalues": pvalues,
            "valid_voxels": np.isfinite(pvalues),
        }
        
        if test == "wilcoxon":
            signed_statistic = np.full(values.shape[1], np.nan)
        
            if eligible.any():
                # Match Wilcoxon's default treatment of zeros and ties.
                magnitudes = np.where(eligible_values == 0, np.nan, np.abs(eligible_values))
                ranks = rankdata(magnitudes, axis=0, method="average", nan_policy="omit")
                ranks = np.nan_to_num(ranks, nan=0.0)
        
                signed_statistic[eligible] = (
                    ranks * np.sign(eligible_values)
                ).sum(axis=0)
        
            entry["signed_statistic"] = signed_statistic
        
        return entry

    if n_jobs == 1 or len(tests) == 1:
        return {test: run_test(test) for test in tests}

    results = _run_jobs(
        run_test,
        tests,
        n_jobs=n_jobs,
        show_progress=show_progress,
        description="One-sample tests",
    )
    return dict(zip(tests, results))


def correct_maps_one_sample(
    stat_results,
    methods=("bonferroni", "fdr_bh"),
    alpha=0.05,
    *,
    cluster=None,
):
    """
    Correct voxelwise p-values and optionally run FSL cluster inference.

    Parameters
    ----------
    stat_results : dict
        Output of ``compute_stats_one_sample``.
    methods : str or sequence of {"bonferroni", "fdr_bh", "fsl_cluster"}
        Correction methods. By default, apply Bonferroni and
        Benjamini-Hochberg.
    alpha : float, default=0.05
        Significance level. FSL cluster inference uses half this value
        per tail.
    cluster : dict, optional
        Required for ``"fsl_cluster"``. Supply ``mask_file``,
        ``z_threshold``, and the same ``group_matrix`` used to compute
        ``stat_results``. The matrix must have one row per independent
        subject and one column per nonzero voxel of the brain mask, in
        SimpleITK array order. The group Z maps and residuals are prepared
        as temporary files. Alternatively, supply ``zstat_file``,
        ``residuals_file``, and ``dof`` in place of ``group_matrix``.

    Returns
    -------
    corrected : dict
        Indexed by test and then method. Voxelwise corrections return
        ``pvalues_corrected``, ``reject``, and ``n_comparisons``.
        The FSL cluster result is under ``corrected["ttest"]`` and
        contains ``reject`` and ``thresholded_z``. All maps are
        one-dimensional and follow the input voxel order.

    Notes
    -----
    Each voxelwise correction uses the finite p-values for one test
    in this call. Repeat calls for different seeds define separate
    correction families.

    With in-memory cluster inputs, voxels with undefined t-tests are
    excluded from the FSL analysis mask and marked as not rejected in
    the returned vector. The file-based mode expects its mask to match
    all t-tested voxels.
    """
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    allowed = ("bonferroni", "fdr_bh", "fsl_cluster")
    if not methods or any(method not in allowed for method in methods):
        raise ValueError("Select bonferroni, fdr_bh, or fsl_cluster.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if "fsl_cluster" in methods and (cluster is None or "ttest" not in stat_results):
        raise ValueError("fsl_cluster requires t-test results and cluster inputs.")

    corrected = _correct_voxelwise(stat_results, methods, alpha)
    if "fsl_cluster" in methods:
        corrected["ttest"]["fsl_cluster"] = _fsl_cluster(
            stat_results["ttest"], alpha, **cluster
        )
    return corrected


def _correct_voxelwise(stat_results, methods, alpha):
    """Apply voxelwise corrections to each test's valid p-values."""
    corrected = {test: {} for test in stat_results}

    for test, entry in stat_results.items():
        pvalues = entry["pvalues"]
        valid = entry["valid_voxels"]
        n_comparisons = int(valid.sum())

        for method in dict.fromkeys(methods):
            if method == "fsl_cluster":
                continue

            adjusted = np.full(pvalues.shape, np.nan)
            reject = np.zeros(pvalues.shape, dtype=bool)

            if n_comparisons:
                rejected, adjusted_p, _, _ = multipletests(
                    pvalues[valid], alpha=alpha, method=method
                )
                reject[valid] = rejected
                adjusted[valid] = adjusted_p

            corrected[test][method] = {
                "pvalues_corrected": adjusted,
                "reject": reject,
                "n_comparisons": n_comparisons,
            }

    return corrected


def _fsl_cluster(
    ttest,
    alpha,
    *,
    mask_file,
    z_threshold,
    group_matrix=None,
    ref_matrix=None,
    zstat_file=None,
    residuals_file=None,
    dof=None,
):
    """Run two-tailed FSL cluster inference for a t-test."""
    from tempfile import TemporaryDirectory

    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("z_threshold must be positive and finite.")

    from_matrix = group_matrix is not None
    if ref_matrix is not None and not from_matrix:
        raise ValueError("ref_matrix requires group_matrix.")
    file_inputs = (zstat_file, residuals_file, dof)
    if from_matrix and any(value is not None for value in file_inputs):
        raise ValueError("Use group_matrix or the precomputed files and dof.")
    if not from_matrix and any(value is None for value in file_inputs):
        raise ValueError("Supply group_matrix or zstat_file, residuals_file, and dof.")

    mask_file = Path(mask_file).resolve()
    mask_img = sitk.ReadImage(str(mask_file))
    if mask_img.GetDimension() != 3:
        raise ValueError("The brain mask must be 3D.")
    mask = sitk.GetArrayFromImage(mask_img) != 0
    n_voxels = int(mask.sum())
    valid_voxels = np.asarray(ttest["valid_voxels"], dtype=bool)
    if valid_voxels.shape != (n_voxels,) or not valid_voxels.any():
        raise ValueError("The brain mask must match the t-test and contain valid voxels.")

    analysis_mask = mask.copy()
    if from_matrix:
        from scipy.special import ndtri_exp

        if ref_matrix is None:
            values = np.asarray(group_matrix, dtype=float)
            if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != n_voxels:
                raise ValueError("Expected group_matrix with at least two rows and one column per mask voxel.")
            matrices = (values[:, valid_voxels],)
            if np.any(np.all(matrices[0] == matrices[0][0], axis=0)):
                raise ValueError("group_matrix must match the data used for the t-test.")
        else:
            matrices = tuple(
                matrix[:, valid_voxels]
                for matrix in _two_sample_matrices(group_matrix, ref_matrix, n_voxels)
            )
            group, ref = matrices
            variable = np.any(group != group[0], axis=0) | np.any(ref != ref[0], axis=0)
            if not variable.all():
                raise ValueError("Both groups have zero within-group variance in a tested voxel.")

        if not all(np.isfinite(matrix).all() for matrix in matrices):
            raise ValueError("The input matrices must be finite at all tested voxels.")
        dof = sum(matrix.shape[0] - 1 for matrix in matrices)
        t_scores = np.asarray(ttest["statistic"])[valid_voxels]
        log_tail = student_t.logsf(np.abs(t_scores), df=dof)
        z_scores = -np.sign(t_scores) * ndtri_exp(log_tail)
        analysis_mask[mask] = valid_voxels
        z = np.zeros(mask.shape, dtype=float)
        z[analysis_mask] = z_scores
    else:
        z_img = sitk.ReadImage(str(zstat_file))
        if not mask_img.IsSameImageGeometryAs(z_img):
            raise ValueError("The mask and Z-statistic image must share a 3D grid.")
        if not valid_voxels.all():
            raise ValueError("The file-based analysis mask must match all t-tested voxels.")
        z = sitk.GetArrayFromImage(z_img).astype(float)
        z_scores = z[analysis_mask]

    if not np.isfinite(z_scores).all():
        raise ValueError("The Z-statistic must be finite within the analysis mask.")

    reject = np.zeros(n_voxels, dtype=bool)
    tails = [
        (tail, sign)
        for tail, sign in (("positive", 1), ("negative", -1))
        if np.any(sign * z_scores >= z_threshold)
    ]
    if not tails:
        return {"reject": reject, "thresholded_z": np.zeros(n_voxels)}

    from nipype.interfaces.fsl import Cluster, SmoothEstimate

    def spatial_image(volume):
        image = sitk.GetImageFromArray(volume)
        image.CopyInformation(mask_img)
        return image

    with TemporaryDirectory() as work_dir:
        work_dir = Path(work_dir)

        if from_matrix:
            mask_file = work_dir / "analysis_mask.nii.gz"
            residuals_file = work_dir / "group_residuals.nii.gz"
            sitk.WriteImage(spatial_image(analysis_mask.astype(np.uint8)), str(mask_file))
            residual_volumes = []
            for matrix in matrices:
                group_mean = matrix.mean(axis=0)
                for subject_values in matrix:
                    volume = np.zeros(mask.shape, dtype=np.float32)
                    volume[analysis_mask] = subject_values - group_mean
                    residual_volumes.append(spatial_image(volume))
            sitk.WriteImage(sitk.JoinSeries(residual_volumes), str(residuals_file))

        smoothness = SmoothEstimate(
            mask_file=str(mask_file),
            residual_fit_file=str(Path(residuals_file).resolve()),
            dof=dof,
        ).run(cwd=str(work_dir)).outputs
        if not np.isfinite(smoothness.dlh) or smoothness.dlh <= 0:
            raise ValueError("FSL smoothest did not return a positive smoothness estimate.")

        for tail, sign in tails:
            tail_img = spatial_image(np.where(analysis_mask, sign * z, 0.0))
            tail_file = work_dir / f"{tail}.nii.gz"
            sitk.WriteImage(tail_img, str(tail_file))

            result = Cluster(
                in_file=str(tail_file),
                threshold=z_threshold,
                pthreshold=alpha / 2,
                dlh=smoothness.dlh,
                volume=smoothness.volume,
                out_threshold_file=str(work_dir / f"{tail}_thresholded.nii.gz"),
                output_type="NIFTI_GZ",
            ).run(cwd=str(work_dir))

            retained = sitk.GetArrayFromImage(
                sitk.ReadImage(str(result.outputs.threshold_file))
            )
            reject |= (retained[mask] > 0) & valid_voxels

    return {
        "reject": reject,
        "thresholded_z": np.where(reject, z[mask], 0.0),
    }

###################################################################################################################################################################

########################################################################### TWO SAMPLEs ###########################################################################

###################################################################################################################################################################

def _two_sample_matrices(group_matrix, ref_matrix, n_voxels=None):
    """Convert and validate two independent subject-by-voxel matrices."""
    group = np.asarray(group_matrix, dtype=float)
    ref = np.asarray(ref_matrix, dtype=float)
    for matrix in (group, ref):
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
            raise ValueError("Each matrix must have at least two subjects and one voxel.")
    if n_voxels is None:
        n_voxels = group.shape[1]
    if group.shape[1] != n_voxels or ref.shape[1] != n_voxels:
        raise ValueError("Both matrices must have the same number of mask voxels.")
    return group, ref


def _pooled_cohens_d(group, ref, axis):
    """Compute the mean difference scaled by pooled sample SD."""
    df_group = group.shape[axis] - 1
    df_ref = ref.shape[axis] - 1
    pooled_sd = np.sqrt(
        (df_group * group.var(axis=axis, ddof=1) + df_ref * ref.var(axis=axis, ddof=1))
        / (df_group + df_ref)
    )
    difference = group.mean(axis=axis) - ref.mean(axis=axis)
    variable = (np.ptp(group, axis=axis) > 0) | (np.ptp(ref, axis=axis) > 0)
    return np.divide(
        difference, pooled_sd, out=np.full_like(difference, np.nan),
        where=(pooled_sd > 0) & variable,
    )


def estimate_maps_two_samples(
    group_matrix, ref_matrix, n_jobs=4, confidence_level=0.95,
    n_bootstrap=2000, chunk_size=128, batch_size=32, seed=0, *,
    show_progress=True,
):
    """
    Estimate voxelwise contrasts between two independent groups.

    Parameters
    ----------
    group_matrix, ref_matrix : array-like of float
        Finite subject-by-voxel matrices with at least two subjects per
        group and the same voxel order. Contrasts are group minus reference.
    n_jobs : int, default=4
        Number of joblib workers for the bootstrap.
    confidence_level : float or None, default=0.95
        Pointwise confidence level. None skips intervals and bootstrapping.
    n_bootstrap : int, default=2000
        Number of bootstrap draws, resampling subjects within each group.
    chunk_size : int, default=128
        Number of voxels per job.
    batch_size : int, default=32
        Number of bootstrap draws processed together by a worker.
    seed : int or None, default=0
        Random seed. Resample indices are shared across voxel chunks.
    show_progress : bool, default=True
        Show progress during parallel bootstrapping.

    Returns
    -------
    result : dict
        Entries mean, median, and cohens_d each contain an estimate
        vector. Mean includes the Welch standard error (se). If requested,
        all entries include ci with shape (n_voxels, 2), lower then upper.
        Median and Cohen's d also include bootstrap se, and Cohen's d
        includes bootstrap_valid_fraction.

    Notes
    -----
    Mean intervals use Welch degrees of freedom. Median differences and
    pooled-SD Cohen's d use percentile bootstrap intervals. If both
    groups have zero sample variance at a voxel, the mean interval and d
    are NaN. Any undefined bootstrap d makes its interval and SE NaN.
    """
    group, ref = _two_sample_matrices(group_matrix, ref_matrix)
    if not (np.isfinite(group).all() and np.isfinite(ref).all()):
        raise ValueError("Both matrices must contain only finite values.")

    n_group, n_voxels = group.shape
    n_ref = ref.shape[0]
    mean = group.mean(axis=0) - ref.mean(axis=0)
    group_var = np.where(np.ptp(group, axis=0) > 0, group.var(axis=0, ddof=1) / n_group, 0)
    ref_var = np.where(np.ptp(ref, axis=0) > 0, ref.var(axis=0, ddof=1) / n_ref, 0)
    variance = group_var + ref_var
    mean_se = np.sqrt(variance)
    result = {
        "mean": {"estimate": mean, "se": mean_se},
        "median": {"estimate": np.median(group, axis=0) - np.median(ref, axis=0)},
        "cohens_d": {"estimate": _pooled_cohens_d(group, ref, axis=0)},
    }
    if confidence_level is None:
        return result
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    for name, value, minimum in (
        ("n_bootstrap", n_bootstrap, 2), ("chunk_size", chunk_size, 1),
        ("batch_size", batch_size, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    denominator = group_var**2 / (n_group - 1) + ref_var**2 / (n_ref - 1)
    dof = np.divide(
        variance**2, denominator, out=np.full(n_voxels, np.nan), where=denominator > 0
    )
    tail = (1 - confidence_level) / 2
    quantiles = (tail, 1 - tail)
    half_width = student_t.ppf(1 - tail, dof) * mean_se
    result["mean"]["ci"] = np.column_stack((mean - half_width, mean + half_width))

    rng = np.random.default_rng(seed)
    group_indices = rng.integers(n_group, size=(n_bootstrap, n_group))
    ref_indices = rng.integers(n_ref, size=(n_bootstrap, n_ref))

    def bootstrap_chunk(start):
        stop = min(start + chunk_size, n_voxels)
        columns = slice(start, stop)
        median_draws = np.empty((n_bootstrap, stop - start))
        d_draws = np.empty_like(median_draws)
        for first in range(0, n_bootstrap, batch_size):
            batch = slice(first, first + batch_size)
            group_draw = group[group_indices[batch], columns]
            ref_draw = ref[ref_indices[batch], columns]
            median_draws[batch] = np.median(group_draw, axis=1) - np.median(ref_draw, axis=1)
            d_draws[batch] = _pooled_cohens_d(group_draw, ref_draw, axis=1)
        return (
            median_draws.std(axis=0, ddof=1),
            np.quantile(median_draws, quantiles, axis=0).T,
            d_draws.std(axis=0, ddof=1),
            np.quantile(d_draws, quantiles, axis=0).T,
            np.isfinite(d_draws).mean(axis=0),
        )

    chunks = _run_jobs(
        bootstrap_chunk, range(0, n_voxels, chunk_size), n_jobs=n_jobs,
        show_progress=show_progress, description="Two-sample bootstrap voxel chunks",
    )
    median_se, median_ci, d_se, d_ci, d_valid_fraction = (
        np.concatenate(parts, axis=0) for parts in zip(*chunks)
    )
    result["median"].update(se=median_se, ci=median_ci)
    result["cohens_d"].update(se=d_se, ci=d_ci, bootstrap_valid_fraction=d_valid_fraction)
    return result


def compute_stats_two_samples(
    group_matrix, ref_matrix, tests=("ttest", "mannwhitneyu", "pooled_ttest"), *, n_jobs=3,
    show_progress=True,
):
    """
    Compute two-sided tests between independent groups at each voxel.

    Parameters
    ----------
    group_matrix, ref_matrix : array-like of float
        Independent subject-by-voxel arrays with the same voxel order and
        at least two subjects per group. Direction is group minus reference.
    tests : str or sequence of {"ttest", "mannwhitneyu", "pooled_ttest"}
        Welch's t-test, Mann-Whitney U, or the equal-variance t-test used
        for optional FSL cluster inference, respectively.
    n_jobs : int, default=1
        Number of workers. Each test receives all its eligible voxels.
    show_progress : bool, default=True
        Show progress for parallel tests.

    Returns
    -------
    results : dict
        Each test returns statistic, pvalues, and valid_voxels vectors.
        Both t-tests also return dof. Mann-Whitney returns U for the
        first group and signed_statistic, the rank-biserial effect
        2U / (n_group * n_ref) - 1. Positive values mean the first group
        tends to rank higher than the reference.

    Notes
    -----
    Nonfinite voxels are excluded. The t-tests also exclude voxels with
    zero variance in both groups.
    """
    group, ref = _two_sample_matrices(group_matrix, ref_matrix)
    tests = (tests,) if isinstance(tests, str) else tuple(tests)
    if not tests or any(test not in ("ttest", "mannwhitneyu", "pooled_ttest") for test in tests):
        raise ValueError("Select 'ttest', 'mannwhitneyu', or 'pooled_ttest'.")
    tests = tuple(dict.fromkeys(tests))
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0
    ):
        raise ValueError("n_jobs must be a nonzero integer.")

    finite = np.isfinite(group).all(axis=0) & np.isfinite(ref).all(axis=0)
    variable = np.any(group != group[0], axis=0) | np.any(ref != ref[0], axis=0)
    ttest_eligible = finite & variable

    def run_test(test):
        statistic = np.full(group.shape[1], np.nan)
        pvalues = np.full_like(statistic, np.nan)
        entry = {"statistic": statistic, "pvalues": pvalues}
        if test == "mannwhitneyu":
            if finite.any():
                result = mannwhitneyu(group[:, finite], ref[:, finite], axis=0)
                statistic[finite], pvalues[finite] = result.statistic, result.pvalue
            entry["signed_statistic"] = 2 * statistic / (len(group) * len(ref)) - 1
        else:
            dof = np.full_like(statistic, np.nan)
            if ttest_eligible.any():
                result = ttest_ind(
                    group[:, ttest_eligible], ref[:, ttest_eligible], axis=0,
                    equal_var=(test == "pooled_ttest"),
                )
                statistic[ttest_eligible] = result.statistic
                pvalues[ttest_eligible] = result.pvalue
                dof[ttest_eligible] = result.df
            entry["dof"] = dof
        entry["valid_voxels"] = np.isfinite(pvalues)
        return entry

    if n_jobs == 1 or len(tests) == 1:
        return {test: run_test(test) for test in tests}
    results = _run_jobs(
        run_test, tests, n_jobs=n_jobs, show_progress=show_progress,
        description="Two-sample tests",
    )
    return dict(zip(tests, results))


def correct_maps_two_samples(
    stat_results, methods=("bonferroni", "fdr_bh", "fsl_cluster"), alpha=0.05, *, cluster=None,
):
    """
    Correct results from independent two-sample tests.

    Parameters
    ----------
    stat_results : dict
        Output of compute_stats_two_samples.
    methods : str or sequence of {"bonferroni", "fdr_bh", "fsl_cluster"}
        Voxelwise corrections apply separately within each test.
        FSL cluster inference is available for pooled_ttest.
    alpha : float, default=0.05
        Significance level. FSL uses half of alpha per tail.
    cluster : dict, optional
        Supply group_matrix, ref_matrix, mask_file, and z_threshold for
        FSL inference. Use the pooled t-test inputs and SimpleITK mask order.

    Returns
    -------
    corrected : dict
        Indexed by test then method. Voxelwise methods return
        pvalues_corrected, reject, and n_comparisons. FSL returns reject
        and thresholded_z under corrected["pooled_ttest"]["fsl_cluster"].
    """
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    allowed = ("bonferroni", "fdr_bh", "fsl_cluster")
    if not methods or any(method not in allowed for method in methods):
        raise ValueError("Select bonferroni, fdr_bh, or fsl_cluster.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    if "fsl_cluster" in methods:
        required = ("group_matrix", "ref_matrix", "mask_file", "z_threshold")
        if (
            "pooled_ttest" not in stat_results or cluster is None
            or any(cluster.get(key) is None for key in required)
        ):
            raise ValueError("FSL requires pooled_ttest, both groups, a mask, and a Z threshold.")

    corrected = _correct_voxelwise(stat_results, methods, alpha)
    if "fsl_cluster" in methods:
        corrected["pooled_ttest"]["fsl_cluster"] = _fsl_cluster(
            stat_results["pooled_ttest"], alpha, **cluster
        )
    return corrected


###################################################################################################################################################################

############################################################################### LMM ###############################################################################

###################################################################################################################################################################

def compute_stats_factorial(
    group_matrix, genotype=None, sex=None, *,
    participants_df=None, factor1="genotype", factor2="sex",
    factor1_order=None, factor2_order=None,
    random_groups=None, n_jobs=1, chunk_size=32,
    show_progress=True, return_intercept=False,
):
    """
    Fit a voxelwise two-by-two factorial model with interaction.

    Parameters
    ----------
    group_matrix : array-like of float
        Matrix of shape (n_observations, n_voxels). Metadata must match
        its row order. Voxels containing nonfinite observations are skipped.
    genotype, sex : array-like of str, optional
        Direct label arrays for the original interface. Pass both arrays
        when participants_df is omitted.
    participants_df : pandas.DataFrame, optional
        Participant metadata in the same positional row order as
        group_matrix. Its index is not used for alignment.
    factor1, factor2 : str, optional
        Names of two distinct binary factor columns in participants_df.
        Defaults are 'genotype' and 'sex'.
    factor1_order, factor2_order : sequence of two levels or None, optional
        Contrast direction for each factor as (positive, negative).
        Each sequence must contain both observed levels exactly once.
        If omitted, levels are sorted alphabetically for strings and
        in ascending order for numbers. The contrast is first minus second.
    random_groups : array-like or None, optional
        One random-intercept group per row, such as litter ID or mouse
        ID for repeated scans. Different groups must be independent.
        Use None for ordinary least squares with independent rows.
    n_jobs : int, default=1
        Joblib workers for mixed-model fitting. Use -1 for all CPUs.
    chunk_size : int, default=32
        Voxels per mixed-model job or per OLS chunk with progress enabled.
    show_progress : bool, default=True
        Show a Rich progress bar for either fitting mode. It counts the
        eligible voxels in completed chunks, including unsuccessful fits.
    return_intercept : bool, default=False
        Include the fixed intercept under the 'intercept' key, with
        the same fields as the other terms. Its estimate equally
        averages the four fitted cell means.

    Returns
    -------
    results : dict
        Entries named by the two factors and 'interaction' each contain estimate,
        se, statistic, pvalues, and valid_voxels arrays of shape
        (n_voxels,). Each main effect is equally averaged over the levels
        of the other factor. The interaction is the factor1 contrast
        at the positive factor2 level minus the same contrast at the
        negative factor2 level.
        The statistic is a t statistic for ordinary least squares and
        a Wald z statistic for the mixed model. P-values are two-sided
        and uncorrected.

    Notes
    -----
    All four combinations of factor levels must be represented. This function
    fits at most one random-intercept grouping factor. Mixed models
    use REML. Convergence failures and singular fits are excluded and
    their estimates and inference are returned as NaN.
    """
    values = np.asarray(group_matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] <= 4 or values.shape[1] == 0:
        raise ValueError("Expected more than four observations and at least one voxel.")
    n_rows, n_voxels = values.shape
    factor_names = (factor1, factor2)
    if return_intercept and "intercept" in factor_names:
        raise ValueError("A factor cannot be named 'intercept' when returning the intercept.")
    if participants_df is None:
        if factor_names != ("genotype", "sex"):
            raise ValueError("Custom factor columns require participants_df.")
        labels_by_factor = (genotype, sex)
    else:
        if genotype is not None or sex is not None:
            raise ValueError("Supply participants_df or direct genotype and sex arrays.")
        if not isinstance(participants_df, pd.DataFrame):
            raise TypeError("participants_df must be a pandas DataFrame.")
        if (
            not all(isinstance(name, str) for name in factor_names)
            or factor1 == factor2 or "interaction" in factor_names
        ):
            raise ValueError("Supply two distinct factor column names other than 'interaction'.")
        missing = [name for name in factor_names if name not in participants_df.columns]
        if missing:
            raise ValueError(f"Missing factor columns: {missing}.")
        labels_by_factor = tuple(participants_df[name] for name in factor_names)

    def encode_factor(name, labels, order):
        labels = np.asarray(labels, dtype=object)
        if labels.shape != (n_rows,) or pd.isna(labels).any():
            raise ValueError(f"Supply one nonmissing {name} label per row.")
        levels = pd.unique(labels)
        if len(levels) != 2:
            raise ValueError(f"Factor {name!r} must have exactly two observed levels.")
        if order is None:
            try:
                positive, _ = sorted(levels)
            except TypeError as error:
                raise ValueError(f"Specify an explicit order for factor {name!r}.") from error
        else:
            try:
                if isinstance(order, (str, bytes)):
                    raise TypeError
                positive, negative = order
                if {positive, negative} != set(levels):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"The order for factor {name!r} must contain both observed levels once."
                ) from error
        return np.where(labels == positive, 0.5, -0.5)

    first = encode_factor(factor1, labels_by_factor[0], factor1_order)
    second = encode_factor(factor2, labels_by_factor[1], factor2_order)
    design = np.column_stack((np.ones(n_rows), first, second, first * second))
    if np.linalg.matrix_rank(design) != 4:
        raise ValueError("All four combinations of factor levels must be represented.")

    candidates = np.flatnonzero(
        np.isfinite(values).all(axis=0) & np.any(values != values[0], axis=0)
    )
    terms = (*factor_names, "interaction")
    if return_intercept:
        terms = ("intercept", *terms)
    coefficient_slice = slice(0 if return_intercept else 1, None)
    estimates = np.full((len(terms), n_voxels), np.nan)
    errors = np.full_like(estimates, np.nan)

    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer.")

    def track_chunks(chunks, description):
        if not show_progress or not candidates.size:
            yield from chunks
            return

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(),
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as progress:
            task = progress.add_task(description, total=candidates.size)
            progress.refresh()
            last_refresh = monotonic()

            for columns, coefficients, standard_errors in chunks:
                yield columns, coefficients, standard_errors
                progress.advance(task, advance=len(columns))
                now = monotonic()
                if now - last_refresh >= 0.5:
                    progress.refresh()
                    last_refresh = now

            progress.refresh()

    if random_groups is None:
        weights = np.linalg.pinv(design)
        weight_norm = np.linalg.norm(weights[coefficient_slice], axis=1)[:, None]
        df = n_rows - design.shape[1]
        eps = np.finfo(float).eps * max(design.shape)

        def fit_ols_chunks():
            step = chunk_size if show_progress else max(1, candidates.size)
            for start in range(0, candidates.size, step):
                columns = candidates[start:start + step]
                y = values[:, columns]
                betas = weights @ y
                residual_norm = np.linalg.norm(y - design @ betas, axis=0)
                usable = residual_norm > eps * np.linalg.norm(y, axis=0)
                sigma = residual_norm / np.sqrt(df)
                yield (
                    columns,
                    np.where(usable, betas[coefficient_slice], np.nan),
                    np.where(usable, weight_norm * sigma, np.nan),
                )

        for columns, coefficients, standard_errors in track_chunks(
            fit_ols_chunks(), "Fitting OLS voxels"
        ):
            estimates[:, columns] = coefficients
            errors[:, columns] = standard_errors
        null = student_t(df=df)
    else:
        clusters = np.asarray(random_groups, dtype=object)
        if clusters.shape != (n_rows,):
            raise ValueError("Supply one random-intercept group per row.")
        codes, levels = pd.factorize(clusters)
        if (codes < 0).any() or not 1 < len(levels) < n_rows:
            raise ValueError("Random intercepts require multiple groups, repeated group labels, and no missing IDs.")
        if (
            isinstance(n_jobs, (bool, np.bool_))
            or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0
        ):
            raise ValueError("n_jobs must be a nonzero integer.")
        def fit_chunk(columns):
            coefficients = np.full((len(terms), len(columns)), np.nan)
            standard_errors = np.full_like(coefficients, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.filterwarnings("ignore", message="Random effects covariance is singular")
                for index, voxel in enumerate(columns):
                    y = values[:, voxel]
                    try:
                        fit = MixedLM(y, design, groups=codes).fit(
                            reml=True, method=["bfgs", "cg"], disp=False
                        )
                        beta = np.asarray(fit.fe_params)[coefficient_slice]
                        se = np.asarray(fit.bse_fe)[coefficient_slice]
                        random_variance = float(np.asarray(fit.cov_re)[0, 0])
                    except (np.linalg.LinAlgError, FloatingPointError):
                        continue
                    if (
                        fit.converged
                        and np.isfinite(fit.scale) and fit.scale > 0
                        and np.isfinite(random_variance)
                        and random_variance > np.finfo(float).eps * fit.scale
                        and np.isfinite(beta).all() and np.isfinite(se).all()
                        and (se > 0).all()
                    ):
                        coefficients[:, index] = beta
                        standard_errors[:, index] = se
            return columns, coefficients, standard_errors

        with Parallel(
            n_jobs=n_jobs, prefer="processes", batch_size=1,
            return_as="generator_unordered",
        ) as parallel:
            completed = parallel(
                delayed(fit_chunk)(candidates[start:start + chunk_size])
                for start in range(0, candidates.size, chunk_size)
            )
            for columns, coefficients, standard_errors in track_chunks(
                completed, "Fitting mixed-model voxels"
            ):
                estimates[:, columns] = coefficients
                errors[:, columns] = standard_errors
        n_failed = int(np.count_nonzero(~np.isfinite(errors[0, candidates])))
        if n_failed:
            warnings.warn(
                f"MixedLM failed or produced a singular fit at {n_failed} of "
                f"{candidates.size} attempted voxels.",
                RuntimeWarning, stacklevel=2,
            )
        null = norm

    statistic = estimates / errors
    pvalues = 2 * null.sf(np.abs(statistic))
    return {
        name: {
            "estimate": estimates[i],
            "se": errors[i],
            "statistic": statistic[i],
            "pvalues": pvalues[i],
            "valid_voxels": np.isfinite(statistic[i]) & np.isfinite(pvalues[i]),
        }
        for i, name in enumerate(terms)
    }
