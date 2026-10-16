"""Tests for rubella.caps, using small random data."""
import numpy as np
import pytest

from rubella.caps import cluster, derivatives, stability
from rubella.caps.main import CAPs

NUM_CLUSTERS = 3


@pytest.fixture
def X():
    """Fake data of 90 samples (e.g. timepoints) by 20 features (e.g. voxels)."""
    return np.random.default_rng(seed=0).standard_normal((90, 20))


@pytest.fixture
def fitted_caps(X):
    return CAPs(n_clusters=NUM_CLUSTERS, n_rep=3, random_state=0).fit_kmeans(X)


def test_pairwise_correlations_matches_numpy(X):
    np.testing.assert_allclose(cluster.pairwise_correlations(X[:5], X[5:8]), np.corrcoef(X[:5], X[5:8])[:5, 5:])


def test_fit_and_predict(fitted_caps, X):
    assert fitted_caps.cluster_centers_.shape == (NUM_CLUSTERS, X.shape[1])
    assert fitted_caps.labels_.shape == (X.shape[0],)
    assert set(fitted_caps.labels_) == set(range(NUM_CLUSTERS))
    # predicting on the fitted data gives back the fitted labels
    np.testing.assert_array_equal(fitted_caps.predict(X), fitted_caps.labels_)


@pytest.mark.parametrize('distance', ['correlation', 'cosine', 'euclidean'])
@pytest.mark.parametrize('centroid_order', ['Cmodes', 'hierarchical', None])
def test_fit_options(X, distance, centroid_order):
    caps = CAPs(n_clusters=NUM_CLUSTERS, distance=distance, centroid_order=centroid_order, random_state=0)
    caps.fit_kmeans(X)
    assert caps.transform(X).shape == (X.shape[0], NUM_CLUSTERS)


def test_fit_is_reproducible(X):
    labels_1 = CAPs(n_clusters=NUM_CLUSTERS, random_state=0).fit_kmeans(X).labels_
    labels_2 = CAPs(n_clusters=NUM_CLUSTERS, random_state=0).fit_kmeans(X).labels_
    np.testing.assert_array_equal(labels_1, labels_2)


def test_estimate_R2(fitted_caps, X):
    r2_per_cluster, r2_overall, recon = fitted_caps.estimate_R2(X)
    assert recon.shape == X.shape
    assert 0 <= r2_overall <= 1


def test_get_caps_probs(fitted_caps, X):
    probs = derivatives.get_caps_probs(X, fitted_caps)
    assert len(probs) == NUM_CLUSTERS
    assert np.isclose(sum(probs), 1)


def test_convert_to_dwell_time():
    assert derivatives.convert_to_dwell_time([1, 1, 2, 2, 2, 3]) == [2, 2, 3, 3, 3, 1]


def test_estimate_transitions():
    labels = np.array([0, 0, 1, 1, 2, 0, 1, 2])
    counts, probability = derivatives.estimate_transitions(labels, k=3)
    assert counts.sum() == len(labels) - 1  # one transition between each pair of consecutive frames
    assert counts[0, 1] == 2  # 0 -> 1 occurs twice
    np.testing.assert_allclose(probability.sum(axis=1), 1)


def test_estimate_transitions_with_censoring():
    censoring_mask = np.array([True, True, False, True, True])
    labels = np.array([0, 1, 1, 2])  # labels of the 4 uncensored frames
    counts, _ = derivatives.estimate_transitions(labels, k=3, censoring_mask=censoring_mask)
    # 0 -> 1 and 1 -> 2 are the only transitions between consecutive uncensored frames
    assert counts.sum() == 2


@pytest.mark.parametrize('censoring_mask', [None, np.ones(8, dtype=bool)])
def test_estimate_transitions_3steps(censoring_mask):
    labels = np.array([0, 0, 1, 1, 2, 0, 1, 2])
    counts, _ = derivatives.estimate_transitions_3steps(labels, k=3, censoring_mask=censoring_mask)
    assert counts.shape == (3, 9)
    assert counts.sum() == len(labels) - 2  # one 3-step sequence per triplet of consecutive frames


def test_stability_report(X):
    k_range = [2, 3]
    caps_l = [CAPs(n_clusters=k, n_rep=3, random_state=0).fit_kmeans(X) for k in k_range]
    stability.stability_report(caps_l, X, k_range)
