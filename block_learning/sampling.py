from libpgm.hybayesiannetwork import HyBayesianNetwork
import numpy as np
import pandas as pd
import math

def generate_synthetics(bn: HyBayesianNetwork, n: int = 1000, evidence: dict = None) -> pd.DataFrame:
    """Function for sampling from BN

    Args:
        bn (HyBayesianNetwork): learnt BN
        n (int, optional): number of samples (rows). Defaults to 1000.
        evidence (dict): dictionary with values of params that initialize nodes

    Returns:
        pd.DataFrame: final sample
    """
    sample = pd.DataFrame()

    if evidence:
        sample = pd.DataFrame(bn.randomsample(5 * n, evidence=evidence))
        # cont_nodes = []
        # for key in bn.nodes.keys():
        #     if (str(type(bn.nodes[key])).split('.')[1] == 'lg') | (str(type(bn.nodes[key])).split('.')[1] == 'lgandd'):
        #         cont_nodes.append(key)
        sample.dropna(inplace=True)
        #sample = sample.loc[(sample.loc[:, cont_nodes].values >= 0).all(axis=1)]
        sample.reset_index(inplace=True, drop=True)
    else:
        sample = pd.DataFrame(bn.randomsample(5 * n))
        # cont_nodes = []
        # for key in bn.nodes.keys():
        #     if (str(type(bn.nodes[key])).split('.')[1] == 'lg') | (str(type(bn.nodes[key])).split('.')[1] == 'lgandd'):
        #         cont_nodes.append(key)
        sample.dropna(inplace=True)
        #sample = sample.loc[(sample.loc[:, cont_nodes].values >= 0).all(axis=1)]
        sample.reset_index(inplace=True, drop=True)
    return sample


def get_probability(sample: pd.DataFrame, initial_data: pd.DataFrame, parameter: str) -> dict:
    """Helper function for calculation probability
       of each label in a sample. Also calculate
       confidence interval for a probability

    Args:
        sample (pd.DataFrame): Data sampled from a bayesian network
        initial_data (pd.DataFrame): Source encoded dataset
        parameter (str): Name of the parameter in which
        we want to calculate probabilities
        of labels

    Returns:
        dict: Dictionary in which
        key - is a label
        value - is a list [lower bound of the interval, probability, higher bound of the interval]
    """
    dict_prob = dict([(str(n), []) for n in initial_data[parameter].unique()])

    for i in dict_prob:
        grouped = sample.groupby(parameter)[parameter].count()
        grouped = {str(key): value for key, value in grouped.items()}
        if i in grouped:
            p = (grouped[i]) / sample.shape[0]
            std = 1.96 * math.sqrt(((1 - p) * p) / sample.shape[0])
            start = p - std
            end = p + std
            dict_prob[i].append(start)
            dict_prob[i].append(p)
            dict_prob[i].append(end)
        else:
            dict_prob[i].append(0)
            dict_prob[i].append(0)
            dict_prob[i].append(0)

    return dict_prob



    

