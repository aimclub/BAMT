import os
import sys
import inspect

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
import numpy as np
import pandas as pd
from copy import copy


def loc_to_DataFrame(data: np.array):
    """Function to convert array to DataFrame
    Args:
        data (np.array): input array

    Returns:
        data (pd.DataFrame): with string columns for filtering
    """
    nodes_type = get_type_numpy(data)
    if data.T.ndim == 1:
        data = data.T
        nodes_type = {0: nodes_type[0]}
    dtype = {
        key: "int64" if value == "disc" else "float64"
        for key, value in nodes_type.items()
        if value in ["disc", "cont"]
    }
    df = pd.DataFrame(data).astype(dtype)
    df.columns = df.columns.map(str)
    return df


def get_type_numpy(data: np.array):
    """Function to define the type of the columns of array
       disc - discrete node
       cont - continuous
    Args:
        data (np.array): input array

    Returns:
        dict: output dictionary where 'key' - node name and 'value' - node type
    Notes:
    -- You may have problems with confusing rows and columns
    """
    arr = data.T

    column_type = {}
    for i, row in enumerate(arr):
        if row.ndim == 0 or row.T.ndim == 0:
            row_is_integer = np.issubdtype(row, np.integer) or row.is_integer()
            column_type[i] = "disc" if row_is_integer else "cont"
        else:
            all_row_is_integer = all(
                np.issubdtype(x, np.integer) or x.is_integer() for x in row
            )
            column_type[i] = "disc" if all_row_is_integer else "cont"
        if column_type[i] not in ["disc", "cont"]:
            print("get_type_numpy: Incorrect type of row")
            print(row)
    return column_type
