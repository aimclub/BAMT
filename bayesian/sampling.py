import pandas as pd

from external.libpgm.hybayesiannetwork import HyBayesianNetwork


def generate_synthetics(bn: HyBayesianNetwork, sign: dict, n: int = 1000, evidence: dict = None) -> pd.DataFrame:
    """Function for sampling from BN

    Args:
        bn (HyBayesianNetwork): learnt BN
        sign (dict): dictionary with nodes signs
        n (int, optional): number of samples (rows). Defaults to 1000.
        evidence (dict): dictionary with values of params that initialize nodes

    Returns:
        pd.DataFrame: final sample
    """
    sample = pd.DataFrame()

    if evidence:
        sample = pd.DataFrame(bn.randomsample(10 * n, evidence=evidence))
        cont_nodes = []
        for key in bn.nodes.keys():
            if (sample[key].dtype == 'float'):
                cont_nodes.append(key)
        for c_keys in cont_nodes:
            if (sign[c_keys] == 'pos'):
                sample = sample.loc[sample[c_keys] >= 0]
        sample.reset_index(inplace=True, drop=True)
    else:
        sample = pd.DataFrame(bn.randomsample(10 * n))
        cont_nodes = []
        for key in bn.nodes.keys():
            if (sample[key].dtype == 'float'):
                cont_nodes.append(key)
        for c_keys in cont_nodes:
            if (sign[c_keys] == 'pos'):
                sample = sample.loc[sample[c_keys] >= 0]
        sample.reset_index(inplace=True, drop=True)
    try:
        sample = sample.sample(n)
    except:
        sample = sample
    sample.reset_index(inplace=True, drop=True)
    return sample
