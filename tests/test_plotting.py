"""Tests for rubella.plotting and rubella.qc, using the fake dataset from conftest.py.

The plots are only checked to run without error, not to look right.
"""
import numpy as np

from rubella import plotting
from rubella.basic import seed_based_FC
from rubella.qc import report, utils


def test_plot_brain(fake_dataset):
    brain_map = np.random.default_rng(seed=0).standard_normal(fake_dataset['num_voxels'])
    for plane in ['sagittal', 'coronal', 'horizontal']:
        fig, ax, cbar = plotting.plot_brain(brain_map, fake_dataset['brain_mask_file'], plane=plane,
                                            slices_per_row=2, plot_cbar=True)
        assert cbar is not None


def test_plot_many_brains(fake_dataset):
    brain_maps = list(np.random.default_rng(seed=0).standard_normal((3, fake_dataset['num_voxels'])))
    fig, ax_list, cbar_list = plotting.plot_many_brains(brain_maps, fake_dataset['brain_mask_file'],
                                                        num_brain_per_row=2)
    assert len(cbar_list) == 3


def test_qc_group_stats(data_obj, seed_roi, fake_dataset):
    SBC_dict = seed_based_FC(data_obj.timeseries_dict, data_obj.brain_mask_file, data_obj.img_id_list, seed_roi)

    tstat, avg, std, percent_overlap = utils.get_group_stats(SBC_dict)
    for group_map in [tstat, avg, std, percent_overlap]:
        assert group_map.shape == (fake_dataset['num_voxels'],)

    dice_list = utils.get_dice_list(SBC_dict, seed_mask_arr=seed_roi)
    assert len(dice_list) == len(SBC_dict)

    report.group_stats_QC_plot(SBC_dict, data_obj.brain_mask_file, template_file=None,
                               selected_slices=[1, 2, 3], slices_per_row=3,
                               qc_data_dict={'fake_motion': [0.1, 0.5, 0.2]})
