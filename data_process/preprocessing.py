from copy import copy
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer


def discretization(data: pd.DataFrame, method: str, columns: list, bins: int = 5) -> pd.DataFrame:
    """Discretization of continuous parameters

    Args:
        data (pd.DataFrame): input dataset
        method (str): discretization approach (equal_intervals, equal_frequency, kmeans)
        columns (list): name of columns for discretization
        bins (int, optional): Number of bins. Defaults to 5.

    Returns:
        pd.DataFrame: output dataset with discretized parameters
    """    
    data = data.dropna()
    data.reset_index(inplace=True, drop=True)
    d_data = copy(data)
    if method == "equal_intervals":
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        data_discrete = est.fit_transform(d_data.loc[:,columns].values)
        d_data[columns] = data_discrete.astype('int')
    elif method == "equal_frequency":
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        data_discrete = est.fit_transform(d_data.loc[:,columns].values)
        d_data[columns] = data_discrete.astype('int')
    elif method == "kmeans":
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
        data_discrete = est.fit_transform(d_data.loc[:,columns].values)
        d_data[columns] = data_discrete.astype('int')
    else:
        print('This discretization method is not supported')
    
    return d_data

def code_categories(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Encoding categorical parameters

    Args:
        data (pd.DataFrame): input dataset
        columns (list): name of categorical columns

    Returns:
        pd.DataFrame: output dataset with encoded parameters
    """    
    data = data.dropna()
    data.reset_index(inplace=True, drop=True)
    d_data = copy(data)
    for column in columns:
        le = preprocessing.LabelEncoder()
        d_data[column] = le.fit_transform(d_data[column].values)
    return d_data


def get_nodes_type (data: pd.DataFrame) -> dict:
    """Function to define the type of the node
       disc - discrete node
       cont - continuous
    Args:
        data (pd.DataFrame): input dataset

    Returns:
        dict: output dictionary where 'key' - node name and 'value' - node type
    """    
    column_type = dict()
    for c in data.columns.to_list():
        if (data[c].dtypes == 'float64') | (data[c].dtypes == 'float32'):
            column_type[c] = 'cont'
        if (data[c].dtypes == 'str') | (data[c].dtypes == 'O') | (data[c].dtypes == 'b'):
            column_type[c] = 'disc'
        if ((data[c].dtypes == 'int64') | (data[c].dtypes == 'int32')):
            column_type[c] = 'disc'
    return column_type