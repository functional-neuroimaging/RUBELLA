"""Tests for rubella.basic_fc, using the fake dataset from conftest.py."""
import numpy as np
import pytest

from rubella import basic_fc
from conftest import IMG_IDS, NUM_FRAMES


def test_seed_based_FC(data_obj, seed_roi, fake_dataset):
    SBC_dict = basic_fc.seed_based_FC(data_obj.timeseries_dict, data_obj.brain_mask_file,
                                      data_obj.img_id_list, seed_roi)
    assert list(SBC_dict.keys()) == IMG_IDS
    for SBC_map in SBC_dict.values():
        assert SBC_map.shape == (fake_dataset['num_voxels'],)
        assert np.all(np.abs(SBC_map) <= 1)


def test_seed_based_FC_rejects_missing_file(data_obj):
    with pytest.raises(FileNotFoundError):
        basic_fc.seed_based_FC(data_obj.timeseries_dict, data_obj.brain_mask_file,
                               data_obj.img_id_list, seed_roi='not_a_file.nii.gz')


def test_compute_dual_regression(data_obj, fake_dataset):
    num_ICs = 4
    IC_maps = np.random.default_rng(seed=0).standard_normal((num_ICs, fake_dataset['num_voxels']))
    DR_dict = basic_fc.compute_dual_regression(data_obj.timeseries_dict, data_obj.brain_mask_file,
                                               data_obj.img_id_list, IC_maps)
    for img_id in IMG_IDS:
        num_frames = data_obj.timeseries_dict[img_id].shape[0]
        assert DR_dict[img_id]['DR_maps'].shape == (fake_dataset['num_voxels'], num_ICs)
        assert DR_dict[img_id]['DR_time'].shape == (num_frames, num_ICs)
        assert DR_dict[img_id]['DR_cross_corr'].shape == (num_ICs, num_ICs)


def test_compute_global_connectivity(data_obj, fake_dataset):
    GC_dict = basic_fc.compute_global_connectivity(data_obj.timeseries_dict, data_obj.img_id_list, verbose=False)
    for GC_map in GC_dict.values():
        assert GC_map.shape == (fake_dataset['num_voxels'],)


def test_compute_FC_matrix(data_obj, fake_dataset):
    # parcellation with 3 parcels of 20 voxels each
    parcellation = np.repeat([1, 2, 3], fake_dataset['num_voxels'] // 3)
    FCM_dict = basic_fc.compute_FC_matrix(data_obj.timeseries_dict, data_obj.brain_mask_file,
                                          data_obj.img_id_list, parcellation)
    for img_id in IMG_IDS:
        matrix = FCM_dict[img_id]['matrix']
        assert FCM_dict[img_id]['parcel_ids'] == [1, 2, 3]
        assert matrix.shape == (3, 3)
        assert np.all(np.isnan(np.diag(matrix)))
        np.testing.assert_allclose(matrix, matrix.T)  # a correlation matrix is symmetric


def test_get_sliding_windows(data_obj, fake_dataset):
    window_length, step_size = 10, 5
    sliding_window_dict = basic_fc.get_sliding_windows(
        data_obj.timeseries_dict, data_obj.img_id_list, censoring_mask_dict=data_obj.censoring_mask,
        window_length=window_length, step_size=step_size, min_censored_window_length=window_length)

    expected_num_windows = (NUM_FRAMES - window_length) // step_size + 1
    for img_id in IMG_IDS:
        windows = sliding_window_dict['sliding_windows'][img_id]
        assert len(windows) == expected_num_windows
        for window in windows:
            assert window is None or window.shape == (window_length, fake_dataset['num_voxels'])
    # the first scan has frames 5-7 censored, so the windows covering them are empty
    assert sliding_window_dict['sliding_windows']['sub-001_ses-1'][0] is None
    assert sliding_window_dict['sliding_windows']['sub-001_ses-1'][1] is None
    assert sliding_window_dict['sliding_windows']['sub-002_ses-1'][0] is not None


def test_get_sliding_windows_with_fc_func(data_obj, seed_roi):
    sliding_window_dict = basic_fc.get_sliding_windows(
        data_obj.timeseries_dict, data_obj.img_id_list, censoring_mask_dict=data_obj.censoring_mask,
        window_length=10, step_size=5,
        fc_func=basic_fc.seed_based_FC,
        fc_kwargs={'brain_mask': data_obj.brain_mask_file, 'seed_roi': seed_roi})
    first_window_FC = sliding_window_dict['sliding_windows']['sub-002_ses-1'][0]
    assert first_window_FC.shape == seed_roi.shape
