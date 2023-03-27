"""
These are functions for dealing with datasets
that have strings as values - as those values
must be converted to integers in order to be
used in many structure learning functions, for
example.

There is, of course, the issue of how to get those
string values back into the underlying representation
after the learning occurs..
"""
import numpy as np


def unique_bins(data):
    """
    Get the unique values for each column in a dataset.
    """
    bins = np.empty(len(data.T), dtype=np.int32)
    i = 0
    for col in data.T:
        bins[i] = len(np.unique(col))
        i += 1
    return bins
