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
