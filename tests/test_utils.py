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
    np.testing.assert_allclose(utils.load_img_to_masked_array(img_file, oriented_mask_file, data_type='continuous'), img_array)

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


def _as_image_format(sitk_img, img_format, file_path):
    """Return a SimpleITK image as a file path, SimpleITK image or nibabel image."""
    if img_format == 'SITK':
        return sitk_img
    sitk.WriteImage(sitk_img, str(file_path))
    if img_format == 'file':
        return file_path
    return nib.load(str(file_path))


@pytest.mark.parametrize('img_format', ['file', 'SITK', 'nibabel'])
def test_load_img_to_masked_array_formats(fake_dataset, tmp_path, img_format):
    timeseries_file = fake_dataset['timeseries_files'][1]
    expected = sitk.GetArrayFromImage(sitk.ReadImage(str(timeseries_file)))[:, fake_dataset['mask_arr']]

    input_img = _as_image_format(sitk.ReadImage(str(timeseries_file)), img_format, tmp_path / 'input.nii.gz')
    brain_mask = _as_image_format(sitk.ReadImage(str(fake_dataset['brain_mask_file'])), img_format, tmp_path / 'mask.nii.gz')
    img_array = utils.load_img_to_masked_array(input_img, brain_mask, data_type='continuous')
    assert img_array.shape == (NUM_FRAMES, fake_dataset['num_voxels'])
    assert img_array.dtype == np.float32  # the float precision of the timeseries is kept
    np.testing.assert_allclose(img_array, expected)


def test_load_img_to_masked_array_data_types(fake_dataset):
    mask_img = sitk.ReadImage(str(fake_dataset['brain_mask_file']))
    labels_arr = fake_dataset['mask_arr'].astype(np.int16)
    labels_arr[2:] *= 2  # two labels, 1 and 2
    labels_img = sitk.GetImageFromArray(labels_arr)
    labels_img.CopyInformation(mask_img)

    mask = utils.load_img_to_masked_array(mask_img, mask_img, data_type='mask')
    assert mask.dtype == bool and mask.all()
    labels = utils.load_img_to_masked_array(labels_img, mask_img, data_type='labels')
    assert labels.dtype == int and set(np.unique(labels)) == {1, 2}
    continuous = utils.load_img_to_masked_array(labels_img, mask_img, data_type='continuous')
    assert continuous.dtype == float and continuous.shape == (fake_dataset['num_voxels'],)


def test_load_img_to_masked_array_rejects_wrong_inputs(fake_dataset):
    mask_file = fake_dataset['brain_mask_file']
    mask_img = sitk.ReadImage(str(mask_file))

    def _img_from_array(arr):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(mask_img)
        return img

    with pytest.raises(ValueError):  # non-binary mask
        utils.load_img_to_masked_array(_img_from_array(fake_dataset['mask_arr'].astype(np.uint8)*255), mask_file, data_type='mask')
    with pytest.raises(ValueError):  # non-integer labels
        utils.load_img_to_masked_array(_img_from_array(fake_dataset['mask_arr']*1.5), mask_file, data_type='labels')
    with pytest.raises(TypeError):  # arrays are not accepted
        utils.load_img_to_masked_array(fake_dataset['mask_arr'], mask_file, data_type='mask')


@pytest.mark.parametrize('data_type', [
    'continuous',
    # resampling a mask calls the ANTs antsApplyTransforms command, which may not be installed
    pytest.param('mask', marks=pytest.mark.skipif(shutil.which('antsApplyTransforms') is None,
                                                  reason="ANTs (antsApplyTransforms) is not installed")),
])
def test_load_img_to_masked_array_resamples(fake_dataset, data_type):
    # an image at twice the resolution of the brain mask, covering the same space, provided as a SimpleITK image
    high_res_arr = fake_dataset['mask_arr'].repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
    high_res_img = sitk.GetImageFromArray(high_res_arr.astype(np.float32))
    high_res_img.SetSpacing((0.5, 0.5, 0.5))
    high_res_img.SetOrigin((-0.25, -0.25, -0.25))

    img_array = utils.load_img_to_masked_array(high_res_img, fake_dataset['brain_mask_file'], data_type=data_type, verbose=False)
    assert img_array.shape == (fake_dataset['num_voxels'],)
    with pytest.raises(ValueError):
        utils.load_img_to_masked_array(high_res_img, fake_dataset['brain_mask_file'], data_type=data_type, resample_to_brain_mask=False)


def test_load_img_to_masked_array_checks_grid(fake_dataset):
    # an image with the same array size as the brain mask, but shifted by one voxel
    mask_img = sitk.ReadImage(str(fake_dataset['brain_mask_file']))
    img = sitk.GetImageFromArray(fake_dataset['mask_arr'].astype(np.float32))
    img.CopyInformation(mask_img)
    img.SetOrigin((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='origin'):
        utils.load_img_to_masked_array(img, mask_img, data_type='continuous', resample_to_brain_mask=False)

    # differences within the float precision of Nifti headers are tolerated
    img.CopyInformation(mask_img)
    img.SetOrigin(tuple(np.array(mask_img.GetOrigin()) + 1e-6))
    utils.load_img_to_masked_array(img, mask_img, data_type='continuous', resample_to_brain_mask=False)

