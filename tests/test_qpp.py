"""Tests for rubella.qpp, using the fake dataset from conftest.py."""
from rubella.basic import get_sliding_windows
from rubella.qpp.main import compute_QPP, count_QPP_occurence


def test_qpp(data_obj, fake_dataset):
    window_length = 10
    # windows with censored frames are left empty, since QPP expects contiguous windows
    sliding_window_dict = get_sliding_windows(
        data_obj.timeseries_dict, data_obj.img_id_list, censoring_mask_dict=data_obj.censoring_mask,
        window_length=window_length, step_size=1, min_censored_window_length=window_length)

    # start from the first window of the second scan, since the first scan's first window is censored
    starting_idx = len(sliding_window_dict['sliding_windows']['sub-001_ses-1'])
    QPP = compute_QPP(sliding_window_dict, window_length, starting_idx=starting_idx)
    assert QPP.shape == (window_length, fake_dataset['num_voxels'])

    QPP_counts = count_QPP_occurence(sliding_window_dict, QPP, remove_consecutives=True)
    assert set(QPP_counts.keys()) == set(data_obj.img_id_list)
