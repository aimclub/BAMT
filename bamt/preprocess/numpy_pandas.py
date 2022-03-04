import os,sys,inspect
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
    if (data.T.ndim == 1):
        data = data.T
        nodes_type = {0: nodes_type[0]}
    dtype = dict()
    for key, value in nodes_type.items():
        if value == 'disc':
            dtype[key] = 'int64'
        if value == 'cont':
            dtype[key] = 'float64'
    df = pd.DataFrame(data)
    df = df.astype(dtype)
    df.columns = df.columns.map(str)
    return df

def get_type_numpy (data: np.array):
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
    
    column_type = dict()
    for i in range(len(arr)):
        if (arr[i].ndim == 0) | (arr[i].T.ndim == 0):
            if np.issubdtype(arr[i], np.integer):
                column_type[i] = 'disc'
            elif arr[i].is_integer():
                column_type[i] = 'disc'
            elif np.issubdtype(arr[i], np.float):
                column_type[i] = 'cont'
            else:
                print('get_type_numpy: Incorrenct type of row')
                print(arr[i])
        else:
            if all(np.issubdtype(x, np.integer) for x in arr[i]):
                column_type[i] = 'disc'
            elif all(x.is_integer() for x in arr[i]):
                column_type[i] = 'disc'
            elif all(np.issubdtype(x, np.float) for x in arr[i]):
                column_type[i] = 'cont'
            else:
                print('get_type_numpy: Incorrenct type of row')
                print(arr[i])
    return column_type