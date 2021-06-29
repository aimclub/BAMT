import operator
from copy import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from external.libpgm.sampleaggregator import SampleAggregator
from preprocess.discretization import get_nodes_type
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from typing import Tuple
from bayesian.train_bn import structure_learning, parameter_learning
from bayesian.save_bn import save_structure, save_params, read_structure, read_params
from sklearn.model_selection import train_test_split


def calculate_acc(bn: HyBayesianNetwork, data: pd.DataFrame, columns: list, method: str, normed: bool = True) -> Tuple[dict, dict, list, list]:
    """Function for calculating of params restoration accuracy

    Args:
        bn (HyBayesianNetwork): fitted BN
        data (pd.DataFrame): test dataset
        columns (list): list of params for restoration
        method (str): method of sampling - simple or mix
        normed (bool): Flag that define type of rmse, if 'false' then raw rmse will be out. Defaults to True.

    Returns:
        dict: accuracy score (discrete vars)
        dict: rmse score (continuous vars)
        list: real data of params
        list: predicted data of params
    """

    accuracy_dict = dict()
    rmse_dict = dict()
    pred_param = [[0 for j in range(data.shape[0])] for i in range(len(columns))]
    real_param = [[0 for j in range(data.shape[0])] for i in range(len(columns))]
    data = data[columns]
    node_type = get_nodes_type(data)
    for i in range(data.shape[0]):
        test = dict(data.iloc[i, :])
        for n, key in enumerate(columns):
            train_dict = copy(test)
            train_dict.pop(key)
            try:
                if node_type[key] == 'disc':
                    agg = SampleAggregator()
                    sample = agg.aggregate(bn.randomsample(2000, method, train_dict))
                    sorted_res = sorted(sample[key].items(), key=operator.itemgetter(1), reverse=True)
                    pred_param[n][i] = sorted_res[0][0]
                    real_param[n][i] = test[key]
                if node_type[key] == 'cont':
                    sample = pd.DataFrame(bn.randomsample(2000, method, train_dict))
                    if (data[key] > 0).any():
                        sample = sample.loc[sample[key] >= 0]
                    if sample.shape[0] == 0:
                        pred_param[n][i] = np.nan
                        real_param[n][i] = np.nan
                    else:
                        pred = np.mean(sample[key].values)
                        pred_param[n][i] = pred
                        real_param[n][i] = test[key]
            except Exception as ex:
                print(ex)
                pred_param[n][i] = np.nan
                real_param[n][i] = np.nan
    for i in range(len(columns)):
        pred_param[i] = [k for k in pred_param[i] if str(k) != 'nan']
        real_param[i] = [k for k in real_param[i] if str(k) != 'nan']
    for n, key in enumerate(columns):
        if node_type[key] == 'disc':
            accuracy_dict[key] = round(accuracy_score(real_param[n], pred_param[n]),2)
        if node_type[key] == 'cont':
            if normed:
                rmse_dict[key] = round(mean_squared_error(real_param[n], pred_param[n], squared=False) / (np.max(real_param[n]) - np.min(real_param[n])), 2)
            else:
                rmse_dict[key] = round(mean_squared_error(real_param[n], pred_param[n], squared=False),2)
    return accuracy_dict, rmse_dict, real_param, pred_param


def LOO_validation(initial_data: pd.DataFrame, data_for_strucure_learning: pd.DataFrame, method: str, columns: list, search: str = 'HC', score: str = 'K2',normed: bool = True) -> Tuple[dict, dict, list, list]:
    """Function for Leave One out cross validation of BN

    Args:
        initial_data (pd.DataFrame): source dataset without coding and discretization.
        data_for_strucure_learning (pd.DataFrame): can be discretized or not depends on what type of structure learning you want. For K2 only discrete can be.
        method (str): method of sampling - simple or mix
        columns (list): list of params for accuracy estimation
        search (str, optional): search strategy for structural learning (HC or evo). Defaults to 'HC'.
        score (str, optional): Score function for HC structure learning. Defaults to 'K2'.
        normed (bool, optional): Flag that define type of rmse, if 'false' then raw rmse will be out. Defaults to True.

    Returns:
        Tuple[dict, dict]: accuracy dict and rmse dict.
    """    
    accuracy_dict = dict()
    rmse_dict = dict()
    node_type = get_nodes_type(initial_data)
    if (score == 'K2') & ('cont' in get_nodes_type(data_for_strucure_learning).values()):
        raise Exception("With K2 function you can use only discrete data for structure learning")
    pred_param = [[0 for j in range(initial_data.shape[0])] for i in range(len(columns))]
    real_param = [[0 for j in range(initial_data.shape[0])] for i in range(len(columns))]
    for i in range (initial_data.shape[0]):
        test = dict(initial_data.iloc[i,:])
        train_data = data_for_strucure_learning.drop(index=i)
        param_train = initial_data.drop(index=i)
        train_data.reset_index(inplace=True, drop = True)
        param_train.reset_index(inplace=True, drop = True)
        bn = structure_learning(train_data, search, node_type, score)
        params = parameter_learning(param_train, node_type, bn, method)
        save_structure(bn, 'LOO_net')
        skel = read_structure('LOO_net')
        save_params(params, 'LOO_net_param')
        params = read_params('LOO_net_param')
        all_bn = HyBayesianNetwork(skel, params)
        for n, key in enumerate(columns):
            train_dict = copy(test)
            train_dict.pop(key)
            try:
                if node_type[key] == 'disc':
                    agg = SampleAggregator()
                    sample = agg.aggregate(all_bn.randomsample(2000, method, train_dict))
                    sorted_res = sorted(sample[key].items(), key=operator.itemgetter(1), reverse=True)
                    pred_param[n][i] = sorted_res[0][0]
                    real_param[n][i] = test[key]
                if node_type[key] == 'cont':
                    sample = pd.DataFrame(all_bn.randomsample(2000, method, train_dict))
                    if (initial_data[key] > 0).any():
                        sample = sample.loc[sample[key] >= 0]
                    if sample.shape[0] == 0:
                        pred_param[n][i] = np.nan
                        real_param[n][i] = np.nan
                    else:
                        pred = np.mean(sample[key].values)
                        pred_param[n][i] = pred
                        real_param[n][i] = test[key]
            except Exception as ex:
                print(ex)
                pred_param[n][i] = np.nan
                real_param[n][i] = np.nan
    for l in range(len(columns)):
        pred_param[l] = [element for element in pred_param[l] if str(element) != 'nan']
        real_param[l] = [element for element in real_param[l] if str(element) != 'nan']
    for n, key in enumerate(columns):
        if node_type[key] == 'disc':
            accuracy_dict[key] = round(accuracy_score(real_param[n], pred_param[n]),2)
        if node_type[key] == 'cont':
            if normed:
                rmse_dict[key] = round(mean_squared_error(real_param[n], pred_param[n], squared=False) / (np.max(real_param[n]) - np.min(real_param[n])), 2)
            else:
                rmse_dict[key] = round(mean_squared_error(real_param[n], pred_param[n], squared=False),2)
    return accuracy_dict, rmse_dict, real_param, pred_param


