from collections import Counter
from copy import copy
from typing import List
from distython import HEOM, HVDM
import gower
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from preprocess.discretization import get_nodes_type, discretization, code_categories

from sklearn.utils.validation import check_consistent_length
from sklearn.metrics._regression import _check_reg_targets

def norm_mean_squared_error(y_true, y_pred,
                       sample_weight=None, multioutput='uniform_average', squared=True):
    """Normalized mean squared error regression loss.


    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors in case of multioutput input.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    squared : bool, default=True
        If True returns NMSE value, if False returns NRMSE value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    output_errors = np.average((y_true - y_pred) ** 2, axis=0,
                               weights=sample_weight)

    if (np.max(y_true)-np.min(y_true)) > 1e-8:
        output_errors /= np.power((np.max(y_true)-np.min(y_true)), 2)
    else:
        output_errors = 0.0

    if not squared:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def acc_rmse_err(acc: dict, rmse: dict):
    count = 0
    sum_err = 0.0
    if bool(acc):
        for value in acc.values():
            sum_err += 1.0 - value
            count += 1
    if bool(rmse):
        for value in rmse.values():
            sum_err += value
            count += 1
    if count > 0:
        return sum_err/count
    else:
        return 0.0

def convert(net1: dict):
    t_net1 = {var: [] for var in net1['V']}
    for e in net1['E']:
        if not t_net1[e[0]]:
            t_net1[e[0]] = [e[1]]
        else:
            t_net1[e[0]] = t_net1[e[0]].append(e[1])
    res_1 = [t_net1[var] for var in t_net1]
    return res_1


def hemm_dist(net1: list, net2: list) -> int:
    sum = 0
    for k in range(len(net1)):
        set1 = set(net1[k] or [])
        set2 = set(net2[k] or [])
        if set1.symmetric_difference(set2) != set():
            sum += len(set1.symmetric_difference(set2))
    return sum

def adjacency_matrix(net_save: list) -> np.array:
    return np.array([[hemm_dist(convert(net_save[i]), convert(net_save[j])) for j in range(len(net_save))] for i in range(len(net_save))])


