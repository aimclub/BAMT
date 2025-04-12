import pytest
import numpy as np
from bamt.utils.gmm_wrapper import GMM


def test_sample_returns_correct_shape():
    X = np.random.randn(100, 3)
    gmm = GMM(n_components=2).from_samples(X)
    n_samples = 10
    samples = gmm.sample(n_samples)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (n_samples, 3)


def test_sample_repeatability_with_seed():
    X = np.random.randn(100, 1)
    gmm1 = GMM(n_components=1, random_state=42).from_samples(X)
    gmm2 = GMM(n_components=1, random_state=42).from_samples(X)

    s1 = gmm1.sample(5)
    s2 = gmm2.sample(5)

    assert np.allclose(s1, s2)


def test_sample_zero_samples():
    X = np.random.randn(50, 2)
    gmm = GMM(n_components=2).from_samples(X)

    samples = gmm.sample(0)
    assert samples.shape == (0, 2)


def test_sample_before_init_raises():
    gmm = GMM(n_components=2)
    with pytest.raises(RuntimeError):
        _ = gmm.sample(5)


def test_sample_single_component():
    X = np.random.randn(100, 1)
    gmm = GMM(n_components=1).from_samples(X)

    samples = gmm.sample(10)
    assert samples.shape == (10, 1)
def test_predict_returns_labels():
    X = np.random.randn(100, 2)
    gmm = GMM(n_components=2).from_samples(X)

    labels = gmm.predict(X)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (100,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.all((0 <= labels) & (labels < gmm.n_components))


def test_predict_zero_samples():
    X = np.random.randn(100, 2)
    gmm = GMM(n_components=2).from_samples(X)

    empty_X = np.empty((0, 2))
    labels = gmm.predict(empty_X)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (0,)
    assert labels.dtype == int


def test_predict_before_init_raises():
    gmm = GMM(n_components=2)
    with pytest.raises(RuntimeError):
        _ = gmm.predict(np.random.randn(5, 2))
def test_predict_proba_shape_and_sum():
    X = np.random.randn(100, 2)
    gmm = GMM(n_components=3).from_samples(X)

    probs = gmm.to_responsibilities(X)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (100, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(100), atol=1e-6)


def test_predict_proba_zero_samples():
    X = np.random.randn(100, 2)
    gmm = GMM(n_components=2).from_samples(X)

    empty_X = np.empty((0, 2))
    probs = gmm.to_responsibilities(empty_X)

    assert probs.shape == (0, 2)


def test_predict_proba_before_init_raises():
    gmm = GMM(n_components=2)
    with pytest.raises(RuntimeError):
        _ = gmm.to_responsibilities(np.random.randn(10, 2))
