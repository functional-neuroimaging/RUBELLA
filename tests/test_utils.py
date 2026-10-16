"""Tests for rubella.utils, using the fake dataset from conftest.py."""
import shutil

import nibabel as nib
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


@pytest.fixture
def oriented_mask_file(fake_dataset):
    """The fake brain mask, with a non-trivial voxel spacing, origin and orientation."""
    mask_img = sitk.ReadImage(str(fake_dataset['brain_mask_file']))
    mask_img.SetSpacing((0.5, 0.6, 0.7))
    mask_img.SetOrigin((1.0, -2.0, 3.0))
    mask_img.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))  # flipped x axis
    mask_file = fake_dataset['folder'] / 'oriented_brain_mask.nii.gz'
    sitk.WriteImage(mask_img, str(mask_file))
    return mask_file


@pytest.mark.parametrize('array_shape', [(60,), (8, 60)])
def test_get_img_from_array_round_trip(oriented_mask_file, tmp_path, array_shape):
    img_array = np.random.default_rng(seed=0).standard_normal(array_shape)
    img = utils.get_img_from_array(img_array, oriented_mask_file)
    assert img.GetDimension() == len(array_shape) + 2  # 1D -> 3D volume, 2D -> 4D image

    # saving the image and loading it back gives the original array
    img_file = tmp_path / 'img.nii.gz'
    sitk.WriteImage(img, str(img_file))
    np.testing.assert_allclose(utils.load_img_to_array(img_file, oriented_mask_file), img_array)

    mask_img = sitk.ReadImage(str(oriented_mask_file))
    np.testing.assert_allclose(sitk.ReadImage(str(img_file)).GetSpacing()[:3], mask_img.GetSpacing())
    np.testing.assert_allclose(sitk.ReadImage(str(img_file)).GetOrigin()[:3], mask_img.GetOrigin())


@pytest.mark.parametrize('array_shape', [(60,), (8, 60)])
def test_get_img_from_array_nibabel_matches_sitk(oriented_mask_file, tmp_path, array_shape):
    img_array = np.random.default_rng(seed=0).standard_normal(array_shape)
    sitk_file, nib_file = tmp_path / 'sitk.nii.gz', tmp_path / 'nib.nii.gz'
    sitk.WriteImage(utils.get_img_from_array(img_array, oriented_mask_file, image_type='SITK'), str(sitk_file))
    nib.save(utils.get_img_from_array(img_array, oriented_mask_file, image_type='nibabel'), str(nib_file))

    # both formats give the same voxel values at the same physical location once saved
    sitk_img, nib_img = sitk.ReadImage(str(sitk_file)), sitk.ReadImage(str(nib_file))
    np.testing.assert_allclose(sitk.GetArrayFromImage(nib_img), sitk.GetArrayFromImage(sitk_img))
    np.testing.assert_allclose(nib_img.GetOrigin()[:3], sitk_img.GetOrigin()[:3])
    np.testing.assert_allclose(nib_img.GetSpacing()[:3], sitk_img.GetSpacing()[:3])
    np.testing.assert_allclose(np.array(nib_img.GetDirection()).reshape(nib_img.GetDimension(), -1)[:3, :3],
                               np.array(sitk_img.GetDirection()).reshape(sitk_img.GetDimension(), -1)[:3, :3])


def test_get_img_from_array_rejects_wrong_inputs(fake_dataset):
    with pytest.raises(ValueError):  # the brain mask has 60 voxels
        utils.get_img_from_array(np.zeros(59), fake_dataset['brain_mask_file'])
    with pytest.raises(ValueError):
        utils.get_img_from_array(np.zeros(60), fake_dataset['brain_mask_file'], image_type='nifti')
