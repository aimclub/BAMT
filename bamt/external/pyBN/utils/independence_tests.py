"""
******************************
Conditional Independence Tests
for Constraint-based Learning
******************************

Implemented Constraint-based Tests
----------------------------------
- mutual information
- Pearson's X^2

I may consider putting this code into its own class structure. The
main benefit I could see from doing this would be the ability to
cache joint/marginal/conditional probabilities for expedited tests.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

from bamt.external.pyBN.utils.data import unique_bins


def mutual_information(data, conditional=False):
    # bins = np.amax(data, axis=0)+1 # read levels for each variable
    bins = unique_bins(data)
    if len(bins) == 1:
        hist, _ = np.histogramdd(data, bins=(bins))  # frequency counts
        Px = (hist / hist.sum()) + (1e-7)
        MI = -1 * np.sum(Px * np.log(Px))
        return round(MI, 4)

    if len(bins) == 2:
        hist, _ = np.histogramdd(data, bins=bins[0:2])  # frequency counts

        Pxy = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Px = np.sum(Pxy, axis=1)  # P(X,Z)
        Py = np.sum(Pxy, axis=0)  # P(Y,Z)

        PxPy = np.outer(Px, Py)
        Pxy += 1e-7
        PxPy += 1e-7
        MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
        return round(MI, 4)
    elif len(bins) > 2 and conditional:
        # CHECK FOR > 3 COLUMNS -> concatenate Z into one column
        if len(bins) > 3:
            data = data.astype('str')
            ncols = len(bins)
            for i in range(len(data)):
                data[i, 2] = ''.join(data[i, 2:ncols])
            data = data.astype(np.int64)[:, 0:3]

        bins = np.amax(data, axis=0)
        hist, _ = np.histogramdd(data, bins=bins)  # frequency counts

        Pxyz = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Pz = np.sum(Pxyz, axis=(0, 1))  # P(Z)
        Pxz = np.sum(Pxyz, axis=1)  # P(X,Z)
        Pyz = np.sum(Pxyz, axis=0)  # P(Y,Z)

        Pxy_z = Pxyz / (Pz + 1e-7)  # P(X,Y | Z) = P(X,Y,Z) / P(Z)
        Px_z = Pxz / (Pz + 1e-7)  # P(X | Z) = P(X,Z) / P(Z)
        Py_z = Pyz / (Pz + 1e-7)  # P(Y | Z) = P(Y,Z) / P(Z)

        Px_y_z = np.empty((Pxy_z.shape))  # P(X|Z)P(Y|Z)
        for i in range(bins[0]):
            for j in range(bins[1]):
                for k in range(bins[2]):
                    Px_y_z[i][j][k] = Px_z[i][k] * Py_z[j][k]
        Pxyz += 1e-7
        Pxy_z += 1e-7
        Px_y_z += 1e-7
        MI = np.sum(Pxyz * np.log(Pxy_z / (Px_y_z)))

        return round(MI, 4)
    elif len(bins) > 2 and conditional == False:
        data = data.astype('str')
        ncols = len(bins)
        for i in range(len(data)):
            data[i, 1] = ''.join(data[i, 1:ncols])
        data = data.astype(np.int64)[:, 0:2]

        hist, _ = np.histogramdd(data, bins=bins[0:2])  # frequency counts

        Pxy = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Px = np.sum(Pxy, axis=1)  # P(X,Z)
        Py = np.sum(Pxy, axis=0)  # P(Y,Z)

        PxPy = np.outer(Px, Py)
        Pxy += 1e-7
        PxPy += 1e-7
        MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
        return round(MI, 4)


def entropy(data):
    """
    In the context of structure learning, and more specifically
    in constraint-based algorithms which rely on the mutual information
    test for conditional independence, it has been proven that the variable
    X in a set which MAXIMIZES mutual information is also the variable which
    MINIMIZES entropy. This fact can be used to reduce the computational
    requirements of tests based on the following relationship:

        Entropy is related to marginal mutual information as follows:
            MI(X;Y) = H(X) - H(X|Y)

        Entropy is related to conditional mutual information as follows:
            MI(X;Y|Z) = H(X|Z) - H(X|Y,Z)

        For one varibale, H(X) is equal to the following:
            -1 * sum of p(x) * log(p(x))

        For two variables H(X|Y) is equal to the following:
            sum over x,y of p(x,y)*log(p(y)/p(x,y))

        For three variables, H(X|Y,Z) is equal to the following:
            -1 * sum of p(x,y,z) * log(p(x|y,z)),
                where p(x|y,z) = p(x,y,z)/p(y)*p(z)
    Arguments
    ----------
    *data* : a nested numpy array
        The data from which to learn - must have at least three
        variables. All conditioned variables (i.e. Z) are compressed
        into one variable.

    Returns
    -------
    *H* : entropy value

    """
    try:
        cols = data.shape[1]
    except IndexError:
        cols = 1

    bins = np.amax(data, axis=0)
    if isinstance(bins, np.ndarray):
        for i in range(len(bins)):
            if bins[i] == 0:
                bins[i] = 1
    else:
        bins = 1
    # bins = unique_bins(data)

    if cols == 1:
        hist, _ = np.histogramdd(data, bins=(bins))  # frequency counts
        Px = hist / hist.sum() + (1e-7)
        H = -1 * np.sum(Px * np.log(Px))

    elif cols == 2:  # two variables -> assume X then Y
        hist, _ = np.histogramdd(data, bins=bins[0:2])  # frequency counts

        Pxy = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Py = np.sum(Pxy, axis=0)  # P(Y)
        Py += 1e-7
        Pxy += 1e-7
        H = np.sum(Pxy * np.log(Py / Pxy))

    else:
        # CHECK FOR > 3 COLUMNS -> concatenate Z into one column
        if cols > 3:
            data = data.astype('str')
            ncols = len(bins)
            for i in range(len(data)):
                data[i, 2] = ''.join(data[i, 2:ncols])
            data = data.astype(np.int64)[:, 0:3]

        bins = np.amax(data, axis=0)
        hist, _ = np.histogramdd(data, bins=bins)  # frequency counts

        Pxyz = hist / hist.sum()  # joint probability distribution over X,Y,Z
        Pyz = np.sum(Pxyz, axis=0)

        Pxyz += 1e-7  # for log -inf
        Pyz += 1e-7
        H = -1 * np.sum(Pxyz * np.log(Pxyz)) + np.sum(Pyz * np.log(Pyz))

    return round(H, 4)
