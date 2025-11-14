"""
Mathematical utilities for BAMT.

This module provides mathematical functions for:
- Gaussian mixture model component selection
- Mixture distribution quantile calculations
- Network comparison metrics (precision, recall, SHD)
"""

import math
from typing import List, Tuple, Dict

import numpy as np
from scipy import stats
from scipy.stats.distributions import chi2
from sklearn.mixture import GaussianMixture


def lrts_comp(data: np.ndarray) -> int:
    """
    Select optimal number of components using Likelihood Ratio Test Statistic.

    Args:
        data: Input data array

    Returns:
        Optimal number of components
    """
    n = 0
    biggest_p = -1 * np.infty
    comp_biggest = 0
    max_comp = 10
    if len(data) < max_comp:
        max_comp = len(data)
    for i in range(1, max_comp + 1, 1):
        gm1 = GaussianMixture(n_components=i, random_state=0)
        gm2 = GaussianMixture(n_components=i + 1, random_state=0)
        gm1.fit(data)
        ll1 = np.mean(gm1.score_samples(data))
        gm2.fit(data)
        ll2 = np.mean(gm2.score_samples(data))
        LR = 2 * (ll2 - ll1)
        p = chi2.sf(LR, 1)
        if p > biggest_p:
            biggest_p = p
            comp_biggest = i
        n = comp_biggest
    return n


def mix_norm_cdf(x: float, weights: List[float], means: List[List[float]],
                 covars: List[List[List[float]]]) -> float:
    """
    Calculate CDF for mixture of normal distributions.

    Args:
        x: Point at which to evaluate CDF
        weights: Component weights
        means: Component means
        covars: Component covariances

    Returns:
        CDF value at x
    """
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * stats.norm.cdf(x, loc=means[i][0], scale=covars[i][0][0])
    return mcdf


def theoretical_quantile(data: np.ndarray, n_comp: int) -> Tuple[List[float], List[float]]:
    """
    Calculate theoretical quantiles for a mixture model.

    Args:
        data: Input data
        n_comp: Number of components

    Returns:
        Tuple of (values, quantiles)
    """
    model = GaussianMixture(n_components=n_comp, random_state=0)
    model.fit(data)
    q = []
    x = []
    step = (np.max(data) - np.min(data)) / 1000
    d = np.arange(np.min(data), np.max(data), step)
    for i in d:
        x.append(i)
        q.append(mix_norm_cdf(i, model.weights_, model.means_, model.covariances_))
    return x, q


def quantile_mix(p: float, vals: List[float], q: List[float]) -> float:
    """
    Find value corresponding to quantile p in mixture distribution.

    Args:
        p: Probability/quantile to find
        vals: Values
        q: Corresponding quantiles

    Returns:
        Value at quantile p
    """
    ind = q.index(min(q, key=lambda x: abs(x - p)))
    return vals[ind]


def probability_mix(val: float, vals: List[float], q: List[float]) -> float:
    """
    Find probability corresponding to value in mixture distribution.

    Args:
        val: Value to find probability for
        vals: Values
        q: Corresponding quantiles/probabilities

    Returns:
        Probability at val
    """
    ind = vals.index(min(vals, key=lambda x: abs(x - val)))
    return q[ind]


def sum_dist(data: np.ndarray, vals: List[float], q: List[float]) -> float:
    """
    Calculate sum of distances between empirical and theoretical quantiles.

    Args:
        data: Empirical data
        vals: Theoretical values
        q: Theoretical quantiles

    Returns:
        Sum of distances
    """
    percs = np.linspace(1, 100, 10)
    x = np.quantile(data, percs / 100)
    y = []
    for p in percs:
        y.append(quantile_mix(p / 100, vals, q))
    dist = 0
    for xi, yi in zip(x, y):
        dist = dist + (abs(-1 * xi + yi)) / math.sqrt(2)
    return dist


def component(data, columns: List[str], method: str) -> int:
    """
    Select optimal number of mixture components using specified method.

    Args:
        data: DataFrame or numpy array
        columns: Column names (for DataFrame) or empty list
        method: Selection method - 'aic', 'bic', 'LRTS', or 'quantile'

    Returns:
        Optimal number of components

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'x': np.random.randn(100)})
        >>> n_comp = component(data, ['x'], 'aic')
    """
    n = 1
    max_comp = 10
    x = []
    if data.shape[0] < max_comp:
        max_comp = data.shape[0]
    if len(columns) == 1:
        x = np.transpose([data[columns[0]].values])
    else:
        x = data[columns].values

    if method == "aic":
        lowest_aic = np.infty
        comp_lowest = 0
        for i in range(1, max_comp + 1, 1):
            gm1 = GaussianMixture(n_components=i, random_state=0)
            gm1.fit(x)
            aic1 = gm1.aic(x)
            if aic1 < lowest_aic:
                lowest_aic = aic1
                comp_lowest = i
            n = comp_lowest

    if method == "bic":
        lowest_bic = np.infty
        comp_lowest = 0
        for i in range(1, max_comp + 1, 1):
            gm1 = GaussianMixture(n_components=i, random_state=0)
            gm1.fit(x)
            bic1 = gm1.bic(x)
            if bic1 < lowest_bic:
                lowest_bic = bic1
                comp_lowest = i
            n = comp_lowest

    if method == "LRTS":
        n = lrts_comp(x)

    if method == "quantile":
        biggest_p = -1 * np.infty
        comp_biggest = 0
        for i in range(1, max_comp, 1):
            vals, q = theoretical_quantile(x, i)
            dist = sum_dist(x, vals, q)
            p = probability_mix(dist, vals, q)
            if p > biggest_p:
                biggest_p = p
                comp_biggest = i
        n = comp_biggest
    return n


def _child_dict(net: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Build child dictionary from edge list.

    Args:
        net: List of (parent, child) edges

    Returns:
        Dictionary mapping children to their parents
    """
    res_dict = dict()
    for e0, e1 in net:
        if e1 in res_dict:
            res_dict[e1].append(e0)
        else:
            res_dict[e1] = [e0]
    return res_dict


def precision_recall(pred_net: List[Tuple[str, str]],
                    true_net: List[Tuple[str, str]],
                    decimal: int = 4) -> Dict[str, float]:
    """
    Calculate precision, recall, and SHD for predicted network.

    Args:
        pred_net: Predicted network edges
        true_net: True network edges
        decimal: Number of decimal places

    Returns:
        Dictionary with metrics: AP, AR, AHP, AHR, SHD

    Example:
        >>> pred = [('A', 'B'), ('B', 'C')]
        >>> true = [('A', 'B'), ('A', 'C')]
        >>> metrics = precision_recall(pred, true)
        >>> print(metrics['SHD'])  # Structural Hamming Distance
    """
    true_dict = _child_dict(true_net)
    corr_undirected = 0
    corr_dir = 0
    for e0, e1 in pred_net:
        flag = True
        if e1 in true_dict:
            if e0 in true_dict[e1]:
                corr_undirected += 1
                corr_dir += 1
                flag = False
        if (e0 in true_dict) and flag:
            if e1 in true_dict[e0]:
                corr_undirected += 1
    pred_len = len(pred_net)
    true_len = len(true_net)
    shd = pred_len + true_len - corr_undirected - corr_dir
    return {
        "AP": round(corr_undirected / pred_len, decimal),
        "AR": round(corr_undirected / true_len, decimal),
        "AHP": round(corr_dir / pred_len, decimal),
        "AHR": round(corr_dir / true_len, decimal),
        "SHD": shd,
    }
