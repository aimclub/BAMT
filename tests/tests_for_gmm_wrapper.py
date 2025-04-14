import pytest
import numpy as np
from bamt.utils.gmm_wrapper import GMM
from gmr import GMM as GMR_GMM


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


def sort_components_by_means(means, covs, priors):
    norms = [np.linalg.norm(m) for m in means]
    indices = np.argsort(norms)
    means_sorted = [means[i] for i in indices]
    covs_sorted = [covs[i] for i in indices]
    priors_sorted = [priors[i] for i in indices]
    return means_sorted, covs_sorted, priors_sorted


def arrays_equal_up_to_sign(a, b, tol=1e-5):
    return np.allclose(a, b, atol=tol) or np.allclose(a, -b, atol=tol)


def gmm_means_close_unordered(m1, m2, tol=1e-5):
    matched = [False] * len(m2)
    for v1 in m1:
        found = False
        for i, v2 in enumerate(m2):
            if not matched[i] and arrays_equal_up_to_sign(v1, v2, tol):
                matched[i] = True
                found = True
                break
        if not found:
            return False
    return all(matched)


def generate_random_positive_def_matrix(dim):
    A = np.random.randn(dim, dim)
    return A @ A.T + dim * np.eye(dim)  # симметричная и положительно определённая


@pytest.mark.parametrize("n_components", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("n_features", [2, 4, 6, 8])
def test_condition_manual_gmm_equivalence(n_components, n_features):
    np.random.seed(n_components * 100 + n_features)  # стабильный сид
    # Случайные параметры
    means = [np.random.randn(n_features).tolist() for _ in range(n_components)]
    covariances = [
        generate_random_positive_def_matrix(n_features).tolist()
        for _ in range(n_components)
    ]

    raw_priors = np.random.rand(n_components)
    priors = (raw_priors / raw_priors.sum()).tolist()

    # Случайный фиксированный вектор
    n_given = min(2, n_features - 1)
    given_indices = sorted(
        np.random.choice(n_features, size=n_given, replace=False).tolist()
    )

    given_values = np.random.randn(1, len(given_indices)).tolist()

    # Создаем модели
    gmr_gmm = GMR_GMM(
        n_components=n_components, means=means, covariances=covariances, priors=priors
    )
    our_gmm = GMM(
        n_components=n_components, means=means, covariances=covariances, priors=priors
    )

    # Условные модели
    gmr_cond = gmr_gmm.condition(given_indices, given_values)
    our_cond = our_gmm.condition(given_indices, given_values)

    # Сравнение
    for m1, m2 in zip(gmr_cond.means, our_cond.means):
        np.testing.assert_allclose(m1, m2, atol=1e-5)

    for c1, c2 in zip(gmr_cond.covariances, our_cond.covariances):
        np.testing.assert_allclose(c1, c2, atol=1e-5)

    np.testing.assert_allclose(gmr_cond.priors, our_cond.priors, atol=1e-5)
