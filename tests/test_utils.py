"""Tests for rubella.utils, using the fake dataset from conftest.py."""
import shutil

import numpy as np
import pytest
import SimpleITK as sitk

from rubella import utils
from conftest import NUM_FRAMES


def test_vcorrcoef_matches_numpy():
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((4, 50))
    Y = rng.standard_normal((3, 50))
    expected = np.corrcoef(X, Y)[:4, 4:]  # cross-correlation block between rows of X and rows of Y
    np.testing.assert_allclose(utils._vcorrcoef(X, Y), expected)


def test_load_img_to_array(fake_dataset):
    timeseries = utils.load_img_to_array(fake_dataset['timeseries_files'][1], fake_dataset['brain_mask_file'])
    assert timeseries.shape == (NUM_FRAMES, fake_dataset['num_voxels'])


@pytest.mark.parametrize('input_is_mask', [
    False,
    # resampling a mask calls the ANTs antsApplyTransforms command, which may not be installed
    pytest.param(True, marks=pytest.mark.skipif(shutil.which('antsApplyTransforms') is None,
                                                reason="ANTs (antsApplyTransforms) is not installed")),
])
def test_load_img_to_array_resamples(fake_dataset, input_is_mask):
    # an image at twice the resolution of the brain mask, covering the same space
    high_res_arr = fake_dataset['mask_arr'].repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    high_res_img = sitk.GetImageFromArray(high_res_arr.astype(np.float32))
    high_res_img.SetSpacing((0.5, 0.5, 0.5))
    high_res_img.SetOrigin((-0.25, -0.25, -0.25))
    high_res_file = fake_dataset['folder'] / f'high_res_{input_is_mask}.nii.gz'
    sitk.WriteImage(high_res_img, str(high_res_file))

    img_array = utils.load_img_to_array(high_res_file, fake_dataset['brain_mask_file'], input_is_mask=input_is_mask)
    assert img_array.shape == (fake_dataset['num_voxels'],)


def test_get_concat_timeseries(data_obj, fake_dataset):
    concat_timeseries = utils.get_concat_timeseries(data_obj.timeseries_dict, data_obj.img_id_list)
    total_frames = sum(ts.shape[0] for ts in data_obj.timeseries_dict.values())
    assert concat_timeseries.shape == (total_frames, fake_dataset['num_voxels'])
