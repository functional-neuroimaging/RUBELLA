"""Shared fake data for the RUBELLA test suite.

pytest automatically loads this file and makes each ``@pytest.fixture`` function below available
to any test that names it as an argument. The fake dataset is tiny (60 brain voxels, 30 frames,
3 scans) so that the whole suite runs in seconds.

Run the tests from the repository root, in an environment where rubella is installed
(``pip install -e .``) along with pytest:

    pytest tests/
"""
import matplotlib
matplotlib.use("Agg")  # draw figures in memory only, without opening windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from rubella.data_obj import DataObject

# dimensions of the fake dataset, kept small so that tests are fast
VOLUME_SHAPE = (5, 6, 7)  # (z, y, x), all different to catch axis mix-ups
NUM_FRAMES = 30           # original number of frames per scan, before censoring
IMG_IDS = ['sub-001_ses-1', 'sub-002_ses-1', 'sub-003_ses-1']
GROUPS = ['sub-001', 'sub-002', 'sub-003']
# frames censored for each scan (only the first scan has censored frames)
CENSORED_FRAMES = {'sub-001_ses-1': [5, 6, 7], 'sub-002_ses-1': [], 'sub-003_ses-1': []}


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory):
    """Write a fake brain mask, fake timeseries and censoring masks as files.

    As with RABIES outputs, censored frames are already removed from the timeseries files,
    and the censoring .csv files keep track of which original frames were removed.
    """
    folder = tmp_path_factory.mktemp("fake_data")
    rng = np.random.default_rng(seed=0)

    # brain mask: a box of 3 x 4 x 5 = 60 voxels inside the volume
    mask_arr = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    mask_arr[1:4, 1:5, 1:6] = 1
    brain_mask_file = folder / "brain_mask.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(mask_arr), str(brain_mask_file))

    timeseries_files = []
    censoring_files = []
    for img_id in IMG_IDS:
        censoring_mask = np.ones(NUM_FRAMES, dtype=bool)
        censoring_mask[CENSORED_FRAMES[img_id]] = False

        timeseries = rng.standard_normal((censoring_mask.sum(),) + VOLUME_SHAPE).astype(np.float32)
        timeseries_file = folder / f"{img_id}_bold.nii.gz"
        # isVector=False is needed so that a 4D array is written as a 4D image, not a 3D vector image
        sitk.WriteImage(sitk.GetImageFromArray(timeseries, isVector=False), str(timeseries_file))
        timeseries_files.append(timeseries_file)

        censoring_file = folder / f"{img_id}_censoring_mask.csv"
        pd.DataFrame({'False = Masked Frames': censoring_mask}).to_csv(censoring_file, index=False)
        censoring_files.append(censoring_file)

    return {
        'folder': folder,
        'brain_mask_file': brain_mask_file,
        'mask_arr': mask_arr.astype(bool),
        'num_voxels': int(mask_arr.sum()),
        'timeseries_files': timeseries_files,
        'censoring_files': censoring_files,
    }


@pytest.fixture
def data_obj(fake_dataset):
    """A DataObject loaded with the fake timeseries and censoring masks."""
    return DataObject(
        fake_dataset['brain_mask_file'],
        timeseries_file_list=fake_dataset['timeseries_files'],
        fname_split='_bold',
        censoring_file_list=fake_dataset['censoring_files'],
        verbose=False,
    )


@pytest.fixture
def seed_roi(fake_dataset):
    """A 1D binary seed mask selecting the first 5 brain voxels."""
    seed = np.zeros(fake_dataset['num_voxels'])
    seed[:5] = 1
    return seed


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test, to free memory."""
    yield
    plt.close('all')
