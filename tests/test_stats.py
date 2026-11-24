"""Regression tests for RUBELLA's seed-map and voxelwise statistics API."""

import numpy as np
import pandas as pd
import SimpleITK as sitk

from rubella.stats import (
    compute_stats_factorial,
    compute_stats_one_sample,
    compute_stats_two_samples,
    correct_maps_one_sample,
    correct_maps_two_samples,
    estimate_maps_one_sample,
    estimate_maps_two_samples,
    perform_SBMs,
)


def test_perform_sbms_returns_fisher_z_maps(data_obj, fake_dataset, tmp_path):
    mask = fake_dataset["mask_arr"]
    roi = np.zeros(mask.shape, dtype=np.uint8)
    seed_voxels = np.arange(5)
    roi[tuple(axis[seed_voxels] for axis in np.where(mask))] = 1
    seed_path = tmp_path / "seed.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(roi), str(seed_path))
    seed_df = pd.DataFrame({"acro": ["seed"], "path": [seed_path]})

    sbms = perform_SBMs(
        data_obj, seed_df, display_fig=False, save_output=False, n_jobs=1,
    )

    assert list(sbms) == ["seed"]
    assert len(sbms["seed"]) == len(data_obj.img_id_list)
    for img_id, actual in zip(data_obj.img_id_list, sbms["seed"]):
        values = data_obj.timeseries_dict[img_id]
        reference = values[:, seed_voxels].mean(axis=1)
        correlations = np.corrcoef(reference, values.T)[0, 1:]
        np.testing.assert_allclose(actual, np.arctanh(correlations), atol=1e-6)


def test_one_sample_estimates_and_corrections():
    group = np.array([[1., -1., 0.], [2., -2., 0.], [3., -3., 0.], [4., -4., 0.]])
    effects = estimate_maps_one_sample(
        group, n_jobs=1, n_bootstrap=20, show_progress=False,
    )
    np.testing.assert_allclose(effects["mean"]["estimate"], [2.5, -2.5, 0.])
    assert effects["mean"]["ci"].shape == (3, 2)
    assert np.isnan(effects["mean"]["ci"][2]).all()

    stats = compute_stats_one_sample(group, n_jobs=1, show_progress=False)
    np.testing.assert_allclose(
        stats["wilcoxon"]["signed_statistic"], [10., -10., np.nan]
    )
    corrected = correct_maps_one_sample(stats, methods=("bonferroni", "fdr_bh"))
    for test in stats:
        entry = corrected[test]["bonferroni"]
        assert entry["n_comparisons"] == 2
        np.testing.assert_allclose(
            entry["pvalues_corrected"][:2],
            np.minimum(2 * stats[test]["pvalues"][:2], 1),
        )
        assert np.isnan(entry["pvalues_corrected"][2])
        assert not entry["reject"][2]
        assert corrected[test]["fdr_bh"]["n_comparisons"] == 2


def test_two_sample_contrasts_keep_group_minus_reference_direction():
    group = np.array([[4., 0.], [6., 1.], [8., 2.]])
    ref = np.array([[0., 4.], [1., 5.], [2., 6.], [3., 7.]])
    effects = estimate_maps_two_samples(group, ref, confidence_level=None)
    np.testing.assert_allclose(effects["mean"]["estimate"], [4.5, -4.5])

    stats = compute_stats_two_samples(
        group, ref, tests=("ttest", "mannwhitneyu", "pooled_ttest"),
        n_jobs=1, show_progress=False,
    )
    np.testing.assert_array_equal(
        np.sign(stats["ttest"]["statistic"]), [1, -1]
    )
    np.testing.assert_allclose(stats["mannwhitneyu"]["signed_statistic"], [1, -1])
    np.testing.assert_allclose(stats["pooled_ttest"]["dof"], [5, 5])
    corrected = correct_maps_two_samples(stats, methods=("fdr_bh",))
    assert corrected["ttest"]["fdr_bh"]["n_comparisons"] == 2


def test_factorial_intercept_and_interaction():
    counts = [2, 3, 4, 5]
    metadata = pd.DataFrame({
        "genotype": np.repeat(["WT", "WT", "Het", "Het"], counts),
        "sex": np.repeat(["M", "F", "M", "F"], counts),
    })
    deviations = np.concatenate([np.arange(n) - (n - 1) / 2 for n in counts])
    values = (np.repeat([1., 3., 4., 9.], counts) + deviations)[:, None]

    stats = compute_stats_factorial(
        values, participants_df=metadata, factor1_order=("Het", "WT"),
        factor2_order=("F", "M"), return_intercept=True, show_progress=False,
    )

    expected = {"intercept": 4.25, "genotype": 4.5, "sex": 3.5, "interaction": 3.}
    assert stats.keys() == expected.keys()
    for name, estimate in expected.items():
        np.testing.assert_allclose(stats[name]["estimate"], [estimate])
        np.testing.assert_array_equal(stats[name]["valid_voxels"], [True])
    corrected = correct_maps_two_samples(stats, methods=("bonferroni",))
    assert corrected["interaction"]["bonferroni"]["n_comparisons"] == 1
