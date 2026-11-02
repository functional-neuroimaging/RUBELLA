"""Tests for rubella.data_obj, using the fake dataset from conftest.py."""
import numpy as np
import pytest
import SimpleITK as sitk

from rubella.data_obj import DataObject
from conftest import CENSORED_FRAMES, GROUPS, IMG_IDS, NUM_FRAMES


def test_load_timeseries(data_obj, fake_dataset):
    assert data_obj.img_id_list == IMG_IDS  # IDs are the filenames cropped at fname_split
    for img_id in IMG_IDS:
        num_retained_frames = NUM_FRAMES - len(CENSORED_FRAMES[img_id])
        assert data_obj.timeseries_dict[img_id].shape == (num_retained_frames, fake_dataset['num_voxels'])


def test_volume_coords(data_obj, fake_dataset):
    volume_coords = data_obj.volume_coords
    mask_arr = fake_dataset['mask_arr']
    assert volume_coords.shape == mask_arr.shape
    assert np.all(np.isnan(volume_coords[~mask_arr]))

    # each brain voxel's coordinate gives the column where that voxel is found in the loaded timeseries
    timeseries_file = fake_dataset['timeseries_files'][1]
    timeseries_4D = sitk.GetArrayFromImage(sitk.ReadImage(str(timeseries_file)))
    timeseries = data_obj.timeseries_dict[data_obj.img_id_list[1]]
    for voxel in zip(*np.where(mask_arr)):
        np.testing.assert_array_equal(timeseries[:, int(volume_coords[voxel])], timeseries_4D[(slice(None),) + voxel])


def test_custom_img_ids(fake_dataset):
    custom_ids = ['a', 'b', 'c']
    data_obj = DataObject(fake_dataset['brain_mask_file'], fake_dataset['timeseries_files'],
                          img_id_list=custom_ids, verbose=False)
    assert list(data_obj.timeseries_dict.keys()) == custom_ids


def test_duplicate_img_ids_raise_error(fake_dataset):
    with pytest.raises(ValueError):
        DataObject(fake_dataset['brain_mask_file'], fake_dataset['timeseries_files'],
                   img_id_list=['a', 'a', 'b'], verbose=False)


def test_censoring_mask(data_obj):
    for img_id in IMG_IDS:
        censoring_mask = data_obj.censoring_mask[img_id]
        assert len(censoring_mask) == NUM_FRAMES
        assert list(np.where(~censoring_mask)[0]) == CENSORED_FRAMES[img_id]


def test_assign_group(data_obj):
    data_obj.assign_group(GROUPS)
    assert data_obj.grouping == dict(zip(IMG_IDS, GROUPS))


def test_load_img_to_masked_array(data_obj, fake_dataset):
    # a 3D image gives a 1D array of brain voxels
    mask_array = data_obj._load_img_to_masked_array(fake_dataset['brain_mask_file'], data_type='mask')
    assert mask_array.shape == (fake_dataset['num_voxels'],)
    assert np.all(mask_array == 1)
