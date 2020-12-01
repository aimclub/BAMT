from libpgm.hybayesiannetwork import HyBayesianNetwork
import numpy as np
import pandas as pd
import math

def generate_synthetics(bn: HyBayesianNetwork, n: int = 100) -> pd.DataFrame:
    """Function for sampling from bayesian network

    Args:
        bn (HyBayesianNetwork): input trained bayesian network
        n (int, optional): Number of samples. Defaults to 100.

    Returns:
        pd.DataFrame: output dataframe with samples.
    """    
    sample = bn.randomsample(n)
    sample_df = pd.DataFrame(sample)
    return sample_df


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
    dict_prob = dict([(n, []) for n in initial_data[parameter].unique()])

    for i in dict_prob:
        grouped = sample.groupby(parameter)[parameter].count()
        if i in grouped:
            p = (sample.groupby(parameter)[parameter].count()[i]) / sample.shape[0]
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



    

