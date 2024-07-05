import sys
import warnings
from copy import copy

import numpy as np
import pandas as pd

from bamt.mi_entropy_gauss import mi_gauss as mutual_information, entropy_all as entropy
from bamt.preprocess.graph import edges_to_dict
from bamt.preprocess.numpy_pandas import get_type_numpy


def info_score(edges: list, data: pd.DataFrame, method="LL"):
    score_funcs = {"LL": log_lik_local, "BIC": BIC_local, "AIC": AIC_local}
    score = score_funcs.get(method.upper(), BIC_local)

    parents_dict = edges_to_dict(edges)
    nodes_with_edges = parents_dict.keys()
    scores = [
        score(data[child_parents].copy(), method)
        for var in nodes_with_edges
        for child_parents in ([var] + parents_dict[var],)
    ]
    scores += [
        score(data[[var]].copy(), method)
        for var in set(data.columns).difference(set(nodes_with_edges))
    ]
    return sum(scores)


##### INFORMATION-THEORETIC SCORING FUNCTIONS #####


def log_likelihood(bn, data, method="LL"):
    """
    Determining log-likelihood of the parameters
    of a Bayesian Network. This is a quite simple
    score/calculation, but it is useful as a straight-forward
    structure learning score.

    Semantically, this can be considered as the evaluation
    of the log-likelihood of the data, given the structure
    and parameters of the BN:
            - log( P( D | Theta_G, G ) )
            where Theta_G are the parameters and G is the structure.

    However, for computational reasons it is best to take
    advantage of the decomposability of the log-likelihood score.

    As an example, if you add an edge from A->B, then you simply
    need to calculate LOG(P'(B|A)) - Log(P(B)), and if the value
    is positive then the edge improves the fitness score and should
    therefore be included.

    Even more, you can expand and manipulate terms to calculate the
    difference between the new graph and the original graph as follows:
            Score(G') - Score(G) = M * I(X,Y),
            where M is the number of data points and I(X,Y) is
            the marginal mutual information calculated using
            the empirical distribution over the data.

    In general, the likelihood score decomposes as follows:
            LL(D | Theta_G, G) =
                    M * Sum over Variables ( I ( X , Parents(X) ) ) -
                    M * Sum over Variables ( H( X ) ),
            where 'I' is mutual information and 'H' is the entropy,
            and M is the number of data points

    Moreover, it is clear to see that H(X) is independent of the choice
    of graph structure (G). Thus, we must only determine the difference
    in the mutual information score of the original graph which had a given
    node and its original parents, and the new graph which has a given node
    and new parents.

    NOTE: This assumes the parameters have already
    been learned for the BN's given structure.

    LL = LL - f(N)*|B|, where f(N) = 0

    Arguments
    ---------
    *bn* : a BayesNet object
            Must have both structure and parameters
            instantiated.
    Notes
    -----
    NROW = data.shape[0]
    mi_score = 0
    ent_score = 0
    for rv in bn.nodes():
            cols = tuple([bn.V.index(rv)].extend([bn.V.index(p) for p in bn.parents(rv)]))
            mi_score += mutual_information(data[:,cols])
            ent_score += entropy(data[:,bn.V.index(rv)])

    return NROW * (mi_score - ent_score)
    """

    NROW = data.shape[0]
    mi_scores = [
        mutual_information(
            data[:, (bn.V.index(rv),) + tuple([bn.V.index(p) for p in bn.parents(rv)])],
            method=method,
        )
        for rv in bn.nodes()
    ]
    ent_scores = [entropy(data[:, bn.V.index(rv)], method=method) for rv in bn.nodes()]
    return NROW * (sum(mi_scores) - sum(ent_scores))


def log_lik_local(data, method="LL"):
    NROW = data.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(data, pd.DataFrame):
            return NROW * (
                mutual_information(data, method=method)
                - entropy(data.iloc[:, 0], method=method)
            )
        elif isinstance(data, pd.Series):
            return 0.0
        elif isinstance(data, np.ndarray):
            return NROW * (
                mutual_information(data, method=method)
                - entropy(data[:, 0], method=method)
            )


def BIC_local(data, method="BIC"):
    NROW = data.shape[0]
    log_score = log_lik_local(data, method=method)
    try:
        penalty = 0.5 * num_params(data) * np.log(NROW)
    except OverflowError as err:
        penalty = sys.float_info.max
    return log_score - penalty


def num_params(data):
    # Convert pandas DataFrame to numpy array
    if isinstance(data, pd.DataFrame):
        data = data.values
    # Convert pandas Series to numpy array
    if isinstance(data, pd.Series):
        data = np.array(copy(data))

    # Calculate number of parameters for numpy array
    if isinstance(data, np.ndarray):
        node_type = get_type_numpy(data)
        columns_for_discrete = [
            param for param, node in node_type.items() if node == "cont"
        ]
        columns_for_code = [
            param for param, node in node_type.items() if node == "disc"
        ]

        prod = 1
        for var in columns_for_code:
            prod *= (
                len(np.unique(data[:, var])) if data.ndim != 1 else len(np.unique(data))
            )
        if columns_for_discrete:
            prod *= len(columns_for_discrete)

        # Handle overflow error
        try:
            return prod
        except OverflowError:
            return sys.float_info.max

    # Raise an error if data type is unexpected
    print("Num_params: Unexpected data type")
    print(data)
    return None


def AIC_local(data, method="AIC"):
    log_score = log_lik_local(data, method=method)
    penalty = num_params(data)
    return log_score - penalty
