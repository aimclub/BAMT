from copy import copy
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple



def get_nodes_sign(data: pd.DataFrame) -> dict:
    """Function to define sign of the node
       neg - if node has negative values
       pos - if node has only positive values

    Args:
        data (pd.DataFrame): input dataset

    Returns:
        dict: output dictionary where 'key' - node name and 'value' - sign of data
    """    
    nodes_types = get_nodes_type(data)
    columns_sign = dict()
    for c in data.columns.to_list():
        if nodes_types[c] == 'cont':
            if (data[c] < 0).any():
                columns_sign[c] = 'neg'
            else:
                columns_sign[c] = 'pos'
    return columns_sign


def get_nodes_type(data: pd.DataFrame) -> dict:
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


def discretization(data: pd.DataFrame, method: str, columns: list, bins: int = 5) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """Discretization of continuous parameters

    Args:
        data (pd.DataFrame): input dataset
        method (str): discretization approach (equal_intervals, equal_frequency, kmeans)
        columns (list): name of columns for discretization
        bins (int, optional): number of bins. Defaults to 5.

    Returns:
        pd.DataFrame: output dataset with discretized parameters
        KBinsDiscretizer: fitted exemplar of discretization class
    """
    data = data.dropna()
    data.reset_index(inplace=True, drop=True)
    d_data = copy(data)
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    if method == "equal_intervals":
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        data_discrete = est.fit_transform(d_data.loc[:, columns].values)
        d_data[columns] = data_discrete.astype('int')
    elif method == "equal_frequency":
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
        data_discrete = est.fit_transform(d_data.loc[:, columns].values)
        d_data[columns] = data_discrete.astype('int')
    elif method == "kmeans":
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
        data_discrete = est.fit_transform(d_data.loc[:, columns].values)
        d_data[columns] = data_discrete.astype('int')
    else:
        raise Exception('This discretization method is not supported')

    return d_data, est


def code_categories(data: pd.DataFrame, method: str, columns: list) -> Tuple[pd.DataFrame, dict]:
    """Encoding categorical parameters

    Args:
        data (pd.DataFrame): input dataset
        method (str): method of encoding (label or onehot)
        columns (list): name of categorical columns

    Returns:
        pd.DataFrame: output dataset with encoded parameters
        dict: dictionary with values and codes
    """
    data = data.dropna()
    data.reset_index(inplace=True, drop=True)
    d_data = copy(data)
    encoder_dict = dict()
    if method == 'label':
        for column in columns:
            le = preprocessing.LabelEncoder()
            d_data[column] = le.fit_transform(d_data[column].values)
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            encoder_dict[column] = mapping
    elif method == 'onehot':
        d_data = pd.get_dummies(d_data, columns=columns)
    else:
        raise Exception('This encoding method is not supported')

    return d_data, encoder_dict


def inverse_discretization(data: pd.DataFrame, columns: list, discretizer: KBinsDiscretizer) -> pd.DataFrame:
    """Inverse discretization for numeric params

    Args:
        data (pd.DataFrame): input dataset with discrete values
        columns (list): colums for inverse_discretization
        discretizer (KBinsDiscretizer): fitted exemplar of discretization class

    Returns:
        pd.DataFrame: output dataset with continuous values
    """
    new_data = copy(data)
    new_data[columns] = discretizer.inverse_transform(new_data[columns].values)

    return new_data


def decode(data: pd.DataFrame, columns: list, encoder_dict: dict) -> pd.DataFrame:
    """Decoding categorical params to initial labels

    Args:
        data (pd.DataFrame): input dataset with encoded params
        columns (list): columns for decoding
        encoder_dict (dict): dictionary with values and codes
    Returns:
        pd.DataFrame: output dataset with decoded params
    """
    for column in columns:
        dict_parameter = encoder_dict[column]
        inv_map = {v: k for k, v in dict_parameter.items()}
        data[column] = data[column].apply(lambda x: inv_map(x))

    return data
