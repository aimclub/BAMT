import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)


from copy import copy
import math
from typing import List
import numpy as np
import pandas as pd
from bamt.external.pyBN.utils.independence_tests import mutual_information, entropy
from bamt.preprocess.discretization import get_nodes_type
from bamt.preprocess.numpy_pandas import loc_to_DataFrame
from bamt.preprocess.graph import edges_to_dict


def query_filter(data: pd.DataFrame, columns: List, values: List):
    """
    Filters the data according to the column-value list
            Arguments
    ----------
    *data* : pandas.DataFrame
    Returns
    -------
    *data_trim* : pandas.DataFrame
        Filtered data.
    Effects
    -------
    None
    """
    data_copy = copy(data)
    filter_str = '`' + str(columns[0]) + '`' + ' == ' + str(values[0])
    if len(columns) == 1:
        return data_copy.query(filter_str)
    else:
        for i in range(1, len(columns)):
            filter_str += ' & ' + '`' + str(columns[i]) + '`' + ' == ' + str(values[i])
        data_trim = data_copy.query(filter_str)
        return data_trim


def entropy_gauss(pd_data):
    """
    Calculate entropy for Gaussian multivariate distributions.
            Arguments
    ----------
    *data* : pd.DataFrame
    Returns
    -------
    *entropy* : a float
        The entropy for Gaussian multivariate distributions.
    Effects
    -------
    None
    """
    if not isinstance(pd_data, pd.Series):
        data = copy(pd_data).values.T 
    else:
        data = np.array(copy(pd_data)).T
    if data.size == 0: 
        return 0.0
    flag_row = False
    flag_col = False
    
    if isinstance(data[0], np.float64):
        flag_row = True
    elif (len(data[0]) < 2) | (data.ndim < 2) :
        flag_row = True
    elif data.shape[0] < 2:
        
        flag_row = True
    if isinstance(copy(data).T[0], np.float64):
        flag_col = True
    elif (len(copy(data).T) < 2) | (copy(data).T.ndim < 2):
        flag_col = True
    elif data.shape[1] < 2:
        
        flag_col = True

    if flag_row & flag_col:
        return sys.float_info.max
    elif flag_row | flag_col:
        var = np.var(data)
        if var > 1e-16:
            return 0.5 * (1 + math.log(var*2*math.pi))
        else:
            return sys.float_info.min
    else:
        var = np.linalg.det(np.cov(data))  
        N = var.ndim      
        if var > 1e-16:
            return 0.5 * (N * (1 + math.log(2*math.pi)) + math.log(var))
        else:
            return sys.float_info.min


def entropy_all(data, method = 'MI'):
    """
        For one varibale, H(X) is equal to the following:
            -1 * sum of p(x) * log(p(x))
        For two variables H(X|Y) is equal to the following:
            sum over x,y of p(x,y)*log(p(y)/p(x,y))
        For three variables, H(X|Y,Z) is equal to the following:
            -1 * sum of p(x,y,z) * log(p(x|y,z)),
                where p(x|y,z) = p(x,y,z)/p(y)*p(z)
    Arguments
    ----------
    *data* : pd.DataFrame
    Returns
    -------
    *H* : entropy value"""
    if type(data) is np.ndarray:
        return entropy_all(loc_to_DataFrame(data), method = method)
    elif isinstance(data, pd.Series):
        return entropy_all(pd.DataFrame(data), method)
    elif (type(data) is pd.DataFrame):
        nodes_type = get_nodes_type(data)
        column_disc = []
        for key in nodes_type:
            if nodes_type[key] == 'disc':
                column_disc.append(key)
        column_cont = []
        for key in nodes_type:
            if nodes_type[key] == 'cont':
                column_cont.append(key)
        data_disc = data[column_disc]
        data_cont = data[column_cont]
        
        if len(column_cont) == 0:
                return entropy(data_disc.values)
        elif len(column_disc) == 0:
            return entropy_gauss(data_cont)
        else:      
            H_disc = entropy(data_disc.values)
            dict_comb = {}
            comb_prob = {}
            for i in range(len(data_disc)):
                row = data_disc.iloc[i]
                comb = ''
                for _, val in row.items():
                    comb = comb + str(val) + ', '
                if not comb in dict_comb:
                    dict_comb[comb] = row
                    comb_prob[comb] = 1
                else:
                    comb_prob[comb] += 1
            
            H_cond = 0.0
            for key in list(dict_comb.keys()):
                filtered_data = query_filter(data, column_disc, list(dict_comb[key]))
                filtered_data = filtered_data[column_cont]
                if comb_prob[key] == 1:
                    if (method == 'BIC') | (method == 'AIC'):
                        H_cond += comb_prob[key] / len(data_disc) * entropy_gauss(data[column_cont])
                    else:
                        H_cond += comb_prob[key] / len(data_disc) * sys.float_info.max
                else:
                    H_cond += comb_prob[key] /len(data_disc) * entropy_gauss(filtered_data)
                if (method == 'BIC') | (method == 'AIC'):
                    if H_cond > entropy_gauss(data[column_cont]):
                        H_cond = entropy_gauss(data[column_cont])
                    
            return (H_disc + H_cond)

def entropy_cond(data, column_cont, column_disc, method):
    data_cont = data[column_cont]
    data_disc = data[column_disc]
    H_gauss = entropy_gauss(data_cont)
    H_cond = 0.0
                
    dict_comb = {}
    comb_prob = {}
    for i in range(len(data_disc)):
        row = data_disc.iloc[i]
        comb = ''
        for _, val in row.items():
            comb = comb + str(val) + ', '
        if not comb in dict_comb:
            dict_comb[comb] = row
            comb_prob[comb] = 1
        else:
            comb_prob[comb] += 1
    
    for key in list(dict_comb.keys()):
        filtered_data = query_filter(data, column_disc, list(dict_comb[key]))
        filtered_data = filtered_data[column_cont]
        if comb_prob[key] == 1:
            if (method == 'BIC') | (method == 'AIC'):
                H_cond += comb_prob[key] / len(data_disc) * H_gauss
            else:
                H_cond += comb_prob[key] / len(data_disc) * sys.float_info.max
        else:
            H_cond += comb_prob[key] / len(data_disc) * entropy_gauss(filtered_data)
    if (method == 'BIC') | (method == 'AIC'):
        if H_cond > H_gauss:
            return H_gauss
        else:
            return H_cond
    return H_cond


def mi_gauss(data, method='MI', conditional=False):
    """
    Calculate Mutual Information based on entropy. 
    In the case of continuous uses entropy for Gaussian multivariate distributions.
            Arguments
    ----------
    *data* : pandas.DataFrame
    Returns
    -------
    *MI* : a float
        The Mutual Information
    Effects
    -------
    None
    Notes
    -----
    - Need to preprocess data with code_categories
    """
    if type(data) is np.ndarray:
        return mi_gauss(loc_to_DataFrame(data), method, conditional)
    elif isinstance(data, pd.Series):
        return(mi_gauss(pd.DataFrame(data)))
    elif type(data) is pd.DataFrame:
        nodes_type = get_nodes_type(data)
        if conditional:
            #Hill-Climbing does not use conditional MI, but other algorithms may require it
            #At the moment it counts on condition of the last row in the list of columns
            print('Warning: conditional == True')
            nodes_type_trim = copy(nodes_type)
            data_trim = copy(data)
            list_keys = list(nodes_type_trim.keys)
            del nodes_type_trim[list_keys[-1]]
            del data_trim[list_keys[-1]]
            return (mi_gauss(data, nodes_type, method) - mi_gauss(data_trim, nodes_type, method))
        else:
            column_disc = []
            for key in nodes_type:
                if nodes_type[key] == 'disc':
                    column_disc.append(key)
            column_cont = []
            for key in nodes_type:
                if nodes_type[key] == 'cont':
                    column_cont.append(key)
            data_disc = data[column_disc]
            data_cont = data[column_cont]

            H_gauss = 0.0
            H_cond = 0.0
            
            if len(column_cont) == 0:
                return(mutual_information(data_disc.values, conditional = False))
            elif len(column_disc) == 0:
                if len(column_cont) == 1:
                    return entropy_gauss(data_cont)
                else:
                    data_last = data_cont[[column_cont[-1]]]
                    column_cont_trim = copy(column_cont)
                    del column_cont_trim[-1]
                    data_cont_trim = data[column_cont_trim]
                
                    H_gauss = entropy_gauss(data_last) + entropy_gauss(data_cont_trim)-entropy_gauss(data_cont)
                    H_gauss = min(H_gauss, entropy_gauss(data_last), entropy_gauss(data_cont_trim))
                    #H_gauss = entropy_gauss(data_cont)
                    H_cond = 0.0
            else:
                H_gauss = entropy_gauss(data_cont)
                H_cond = entropy_cond(data, column_cont, column_disc, method)
                
            return(H_gauss-H_cond)

def mi(edges: list, data: pd.DataFrame, method='MI'):
    """
    Bypasses all nodes and summarizes scores, 
    taking into account the parent-child relationship.
            Arguments
    ----------
    *edges* : list
    *data* : pd.DataFrame
    Returns
    -------
    *sum_score* : float
    Effects
    -------
    None
    """
    parents_dict = edges_to_dict(edges)
    sum_score = 0.0
    nodes_with_edges = parents_dict.keys()
    for var in nodes_with_edges:
        child_parents = [var]
        child_parents.extend(parents_dict[var])
        sum_score += mi_gauss(copy(data[child_parents]), method)
    nodes_without_edges = list(set(data.columns).difference(set(nodes_with_edges)))
    for var in nodes_without_edges:
        sum_score += mi_gauss(copy(data[var]), method)
    return sum_score