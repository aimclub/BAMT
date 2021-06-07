from collections import Counter
from copy import copy
from typing import List
from distython import HEOM, HVDM
import gower
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from preprocess.discretization import get_nodes_type, discretization, code_categories


def get_subsample(target_object: dict, data: pd.DataFrame, method: str, list_of_params: list,
                number_of_top: int, percent_of_interval: float = 10, weights: list = None) -> pd.DataFrame:
    """Function for getting analogs of target unit

    Args:
        target_object (dict): params of target object
        data (pd.DataFrame): dataset for searching of similar
        method (str): metric for getting similar (filtering, gower, cosine, hamming)
        list_of_params (list): parameters by which similar are searched
        number_of_top (int): how many closest similar to display
        percent_of_interval (float, optional): hit interval range for continuous parameters (only for 'filtering'). Defaults to 10.
        weights (list, optional): list of weights for each parameter. Defaults to None.

    Raises:
        Exception: if unsupported metric specified

    Returns:
        pd.DataFrame: dataset of similar objects
    """
    analogs = pd.DataFrame()
    init_data = copy(data)
    init_target_unit = copy(target_object)

    index_of_target = -1
    comb = init_target_unit.values()
    mask = np.full(len(data), True)
    for col, val in zip(init_target_unit.keys(), comb):
        mask = (mask) & (data[col] == val)
    find_target = data[mask]
    if find_target.shape[0] == 1:
        index_of_target = find_target.index.tolist()[0]
    elif find_target.shape[0] == 0:
        init_data = pd.concat([init_data, pd.DataFrame(init_target_unit, index=[init_data.shape[0]])])
        index_of_target = init_data.shape[0] - 1
    else:
        index_of_target = find_target.index.tolist()[0]
    data = init_data[list_of_params]
    # for param in list_of_params:
    #     if param not in target_unit:
    #         target_unit.pop(param)

    if method == 'filtering':
        result = analog_finder2(data, index_of_target, list_of_params, number_of_top + 1, percent_of_interval)
        indexes = []
        for i in range(number_of_top + 1):
            indexes.append(result[0][i][0])
        indexes = indexes[1:]
        analogs = init_data.loc[indexes, :]
    elif method == 'gower':
        x = data.values
        dists = []
        if weights:
            dists = gower.gower_matrix(x, weight=np.array(weights))
        else:
            dists = gower.gower_matrix(x)
        indexes = np.argsort(dists[index_of_target])
        cluster_index = indexes[1:number_of_top + 1]
        analogs = init_data.loc[cluster_index, :]
    elif method == 'cosine':
        if weights:
            node_type = get_nodes_type(data)
            cont_param = []
            disc_param = []
            new_weights = []
            for param, w in zip(list_of_params, weights):
                if node_type[param] == 'cont':
                    new_weights.append(w)
                    cont_param.append(param)
                else:
                    disc_param.append(param)
                    for j in range(len(data[param].unique())):
                        new_weights.append(w)
            X, code_dict = code_categories(data, 'onehot', disc_param)
            X[cont_param] = X[cont_param].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            X = X.values
            dists = cdist([X[index_of_target, :]], X[:, :], metric='cosine', w=np.array(new_weights))
            indexes = np.argsort(dists[0])
            cluster_index = indexes[1:number_of_top + 1]
            analogs = init_data.loc[cluster_index, :]
        else:
            node_type = get_nodes_type(data)
            cont_param = []
            disc_param = []
            for param in list_of_params:
                if node_type[param] == 'cont':
                    cont_param.append(param)
                else:
                    disc_param.append(param)
            x, code_dict = code_categories(data, 'onehot', disc_param)
            x[cont_param] = x[cont_param].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            x = x.values
            dists = cdist([x[index_of_target, :]], x[:, :], metric='cosine')
            indexes = np.argsort(dists[0])
            cluster_index = indexes[1:number_of_top + 1]
            analogs = init_data.loc[cluster_index, :]
    elif method == 'hamming':
        node_type = get_nodes_type(data)
        cont_param = []
        disc_param = []
        for param in list_of_params:
            if node_type[param] == 'cont':
                cont_param.append(param)
            else:
                disc_param.append(param)
        hamming_data, est = discretization(data, 'kmeans', cont_param)
        x, code_dict = code_categories(hamming_data, 'onehot', list_of_params)
        x = x.values
        dists = cdist([x[index_of_target, :]], x[:, :], metric='hamming')
        indexes = np.argsort(dists[0])
        cluster_index = indexes[1:number_of_top + 1]
        analogs = init_data.loc[cluster_index, :]
    elif method == 'HEOM':
        node_type = get_nodes_type(data)
        cont_param = []
        disc_param = []
        for param in list_of_params:
            if node_type[param] == 'cont':
                cont_param.append(param)
            else:
                disc_param.append(param)
        columns = data.columns.to_list()
        cat_index = [columns.index(x) for x in disc_param]
        code_data, coder = code_categories(data, 'label', disc_param)
        x = code_data.values
        heom_metric = HEOM(x, cat_ix=cat_index)
        result_metrics = np.full(len(x), 0)
        for i in range(data.shape[0]):
            result_metrics[i] = heom_metric.heom(x[index_of_target], x[i])
        indexes = np.argsort(result_metrics)
        cluster_index = indexes[1:number_of_top + 1]
        analogs = init_data.loc[cluster_index, :]
    elif method == 'HVDM':
        node_type = get_nodes_type(data)
        cont_param = []
        disc_param = []
        for param in list_of_params:
            if node_type[param] == 'cont':
                cont_param.append(param)
            else:
                disc_param.append(param)
        columns = data.columns.to_list()
        cat_index = [columns.index(x) for x in disc_param]
        code_data, coder = code_categories(data, 'label', disc_param)
        x = code_data.values
        hvdm_metric = HVDM(x, y_ix = [[cat_index]], cat_ix=cat_index)
        result_metrics = np.full(len(x), 0)
        for i in range(data.shape[0]):
            result_metrics[i] = hvdm_metric.hvdm(x[index_of_target], x[i])
        indexes = np.argsort(result_metrics)
        cluster_index = indexes[1:number_of_top + 1]
        analogs = init_data.loc[cluster_index, :]
    else:
        raise Exception("This type of search for analogs is not supported")
    return analogs


def filtering2(data: pd.DataFrame, parameters: List):
    """[summary]

    Args:
        data (pd.DataFrame): [description]
        parameters (List): [description]

    Returns:
        [type]: [description]
    """
    length_of_param = len(parameters)
    columns = data.columns.to_list()
    cat = data.columns[data.dtypes == object].to_list()
    # cont=[x for x in columns if x not in cat]
    fdata = data.copy()
    for params_range_ind in range(length_of_param):
        if parameters[params_range_ind][0] in cat:
            val = parameters[params_range_ind][1]
            if type(val) == list and len(val) > 1:
                val = list(set([item for sublist in [i.split('/') for i in val] for item in sublist]))

            elif type(val) == list and len(val) == 1:
                val = val[0].split('/')

            # if '/' in parameters[params_range_ind][1]:
            #    parameters[params_range_ind][1]= parameters[params_range_ind][1].split('/')
            # print(parameters[params_range_ind][1])
            # fdata = fdata[fdata[parameters[params_range_ind][0]].isin(parameters[params_range_ind][1])]
            # print(val)
            fdata = fdata[fdata[parameters[params_range_ind][0]].str.contains('|'.join(val), na=False)]
        else:
            number_of_ranges = len(parameters[params_range_ind][1]) / 2
            qry = str(parameters[params_range_ind][1][0]) + "<= `" + parameters[params_range_ind][0] + "`<= " + str(
                parameters[params_range_ind][1][1])
            number_of_ranges = int(number_of_ranges)
            if number_of_ranges > 1:
                for number_of_ranges_ind in range(number_of_ranges - 1):
                    qry = qry + "|" + str(parameters[params_range_ind][1][(number_of_ranges_ind + 1) * 2]) + "<= `" + \
                          parameters[params_range_ind][0] + "`<= " + str(
                        parameters[params_range_ind][1][(number_of_ranges_ind + 1) * 2 + 1])
            fdata = fdata.query(qry)
    return fdata


def analog_finder2(data, target_unit, list_of_filter_param, number_of_top, percent):
    """[summary]

    Args:
        data ([type]): [description]
        target_unit ([type]): [description]
        list_of_filter_param ([type]): [description]
        number_of_top ([type]): [description]
        percent ([type]): [description]

    Returns:
        [type]: [description]
    """
    percent = percent / 100
    target = data.loc[target_unit]
    if list_of_filter_param == []:
        target_columns = data.columns
    else:
        target_columns = list_of_filter_param
    target_columns_cat = data.columns[data.dtypes == object].to_list()
    target_columns_cont = data.columns[data.dtypes != object].to_list()
    target_columns_cat = [i for i in target_columns_cat if i in target_columns]
    target_columns_cont = [i for i in target_columns_cont if i in target_columns]
    len_target_columns = len(target_columns)

    target_val_cat = [[iter1, [target[iter1]]] for iter1 in target_columns_cat]
    target_val_cont = [[iter1, [target[iter1], target[iter1]]] for iter1 in target_columns_cont]
    target_val_cont2 = [
        [t1[0], [t1[1][0] - percent * t1[1][0], (1 + percent) * t1[1][0]]] if int(t1[1][0]) > 0 else [t1[0], [
            (1 + percent) * t1[1][0], t1[1][0] - percent * t1[1][0]]] for t1 in target_val_cont]

    arrlist_cat = [filtering2(data, [iter1]) for iter1 in target_val_cat]
    arrlist_cont = [filtering2(data, [iter1]) for iter1 in target_val_cont2]
    arrlist = arrlist_cat + arrlist_cont
    arrindex = [i.index.to_list() for i in arrlist]

    flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]

    arrindex_flat = flatten(arrindex)
    arrindex_conted = Counter(arrindex_flat).most_common()

    top_n = arrindex_conted[0:number_of_top]
    list_top_index = [i[0] for i in top_n]
    top_rows = data.loc[list_top_index]

    return top_n, top_rows