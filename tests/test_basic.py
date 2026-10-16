"""Tests for rubella.basic, using the fake dataset from conftest.py."""
import numpy as np
import pytest
import SimpleITK as sitk

from rubella import basic
from conftest import IMG_IDS, NUM_FRAMES


def test_vcorrcoef_matches_numpy():
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((4, 50))
    Y = rng.standard_normal((3, 50))
    expected = np.corrcoef(X, Y)[:4, 4:]  # cross-correlation block between rows of X and rows of Y
    np.testing.assert_allclose(basic.vcorrcoef(X, Y), expected)


def test_load_img_to_array(fake_dataset):
    timeseries = basic.load_img_to_array(fake_dataset['timeseries_files'][1], fake_dataset['brain_mask_file'])
    assert timeseries.shape == (NUM_FRAMES, fake_dataset['num_voxels'])


@pytest.mark.parametrize('input_is_mask', [False, True])
def test_load_img_to_array_resamples(fake_dataset, input_is_mask):
    # an image at twice the resolution of the brain mask, covering the same space
    high_res_arr = fake_dataset['mask_arr'].repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    high_res_img = sitk.GetImageFromArray(high_res_arr.astype(np.float32))
    high_res_img.SetSpacing((0.5, 0.5, 0.5))
    high_res_img.SetOrigin((-0.25, -0.25, -0.25))
    high_res_file = fake_dataset['folder'] / f'high_res_{input_is_mask}.nii.gz'
    sitk.WriteImage(high_res_img, str(high_res_file))

    img_array = basic.load_img_to_array(high_res_file, fake_dataset['brain_mask_file'], input_is_mask=input_is_mask)
    assert img_array.shape == (fake_dataset['num_voxels'],)


def test_seed_based_FC(data_obj, seed_roi, fake_dataset):
    SBC_dict = basic.seed_based_FC(data_obj.timeseries_dict, data_obj.brain_mask_file,
                                   data_obj.img_id_list, seed_roi)
    assert list(SBC_dict.keys()) == IMG_IDS
    for SBC_map in SBC_dict.values():
        assert SBC_map.shape == (fake_dataset['num_voxels'],)
        assert np.all(np.abs(SBC_map) <= 1)


def test_seed_based_FC_rejects_missing_file(data_obj):
    with pytest.raises(ValueError):
        basic.seed_based_FC(data_obj.timeseries_dict, data_obj.brain_mask_file,
                            data_obj.img_id_list, seed_roi='not_a_file.nii.gz')


def test_compute_dual_regression(data_obj, fake_dataset):
    num_ICs = 4
    IC_maps = np.random.default_rng(seed=0).standard_normal((num_ICs, fake_dataset['num_voxels']))
    DR_dict = basic.compute_dual_regression(data_obj.timeseries_dict, data_obj.brain_mask_file,
                                            data_obj.img_id_list, IC_maps)
    for img_id in IMG_IDS:
        num_frames = data_obj.timeseries_dict[img_id].shape[0]
        assert DR_dict[img_id]['DR_maps'].shape == (fake_dataset['num_voxels'], num_ICs)
        assert DR_dict[img_id]['DR_time'].shape == (num_frames, num_ICs)
        assert DR_dict[img_id]['DR_cross_corr'].shape == (num_ICs, num_ICs)


def test_compute_global_connectivity(data_obj, fake_dataset):
    GC_dict = basic.compute_global_connectivity(data_obj.timeseries_dict, data_obj.img_id_list, verbose=False)
    for GC_map in GC_dict.values():
        assert GC_map.shape == (fake_dataset['num_voxels'],)


def test_compute_FC_matrix(data_obj, fake_dataset):
    # parcellation with 3 parcels of 20 voxels each
    parcellation = np.repeat([1, 2, 3], fake_dataset['num_voxels'] // 3)
    FCM_dict = basic.compute_FC_matrix(data_obj.timeseries_dict, data_obj.brain_mask_file,
                                       data_obj.img_id_list, parcellation)
    for img_id in IMG_IDS:
        matrix = FCM_dict[img_id]['matrix']
        assert FCM_dict[img_id]['parcel_ids'] == [1, 2, 3]
        assert matrix.shape == (3, 3)
        assert np.all(np.isnan(np.diag(matrix)))
        np.testing.assert_allclose(matrix, matrix.T)  # a correlation matrix is symmetric


def test_get_concat_timeseries(data_obj, fake_dataset):
    concat_timeseries = basic.get_concat_timeseries(data_obj.timeseries_dict, data_obj.img_id_list)
    total_frames = sum(ts.shape[0] for ts in data_obj.timeseries_dict.values())
    assert concat_timeseries.shape == (total_frames, fake_dataset['num_voxels'])


def test_get_sliding_windows(data_obj, fake_dataset):
    window_length, step_size = 10, 5
    sliding_window_dict = basic.get_sliding_windows(
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
    sliding_window_dict = basic.get_sliding_windows(
        data_obj.timeseries_dict, data_obj.img_id_list, censoring_mask_dict=data_obj.censoring_mask,
        window_length=10, step_size=5,
        fc_func=basic.seed_based_FC,
        fc_kwargs={'brain_mask_file': data_obj.brain_mask_file, 'seed_roi': seed_roi})
    first_window_FC = sliding_window_dict['sliding_windows']['sub-002_ses-1'][0]
    assert first_window_FC.shape == seed_roi.shape
