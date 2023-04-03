import random

from pandas import DataFrame

from .base import BaseNode
from .schema import DiscreteParams
from typing import Type, Dict, Union, List

from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
from concurrent.futures import ThreadPoolExecutor


class DiscreteNode(BaseNode):
    """
    Main class of Discrete Node
    """

    def __init__(self, name):
        super(DiscreteNode, self).__init__(name)
        self.type = 'Discrete'

    def fit_parameters(self, data: DataFrame, num_workers: int = 1):
        """
        Train params for Discrete Node
        data: DataFrame to train on
        num_workers: number of Parallel Workers
        Method returns probas dict with the following format {[<combinations>: value]}
        and vals, list of appeared values in combinations
        """

        def worker(node: Type[BaseNode]) -> DiscreteParams:
            parents = node.disc_parents + node.cont_parents
            if not parents:
                dist = DiscreteDistribution.from_samples(
                    data[node.name].values)
                cprob = list(dict(sorted(dist.items())).values())
                vals = sorted([str(x)
                               for x in list(dist.parameters[0].keys())])
            else:
                dist = DiscreteDistribution.from_samples(
                    data[node.name].values)
                vals = sorted([str(x)
                               for x in list(dist.parameters[0].keys())])
                dist = ConditionalProbabilityTable.from_samples(
                    data[parents + [node.name]].values)
                params = dist.parameters[0]
                cprob = dict()
                for i in range(0, len(params), len(vals)):
                    probs = []
                    for j in range(i, (i + len(vals))):
                        probs.append(params[j][-1])
                    combination = [str(x) for x in params[i][0:len(parents)]]
                    cprob[str(combination)] = probs
            return {"cprob": cprob, 'vals': vals}

        pool = ThreadPoolExecutor(num_workers)
        future = pool.submit(worker, self)
        return future.result()

    @staticmethod
    def choose(node_info: Dict[str, Union[float, str]],
               pvals: List[str]) -> str:
        """
        Return value from discrete node
        params:
        node_info: nodes info from distributions
        pvals: parent values
        """
        rindex = 0
        random.seed()
        vals = node_info['vals']
        if not pvals:
            dist = node_info['cprob']
        else:
            # noinspection PyTypeChecker
            dist = node_info['cprob'][str(pvals)]
        lbound = 0
        ubound = 0
        rand = random.random()
        for interval in range(len(dist)):
            ubound += dist[interval]
            if lbound <= rand < ubound:
                rindex = interval
                break
            else:
                lbound = ubound

        return vals[rindex]

    @staticmethod
    def predict(node_info: Dict[str, Union[float, str]],
                pvals: List[str]) -> str:
        """function for prediction based on evidence values in discrete node

        Args:
            node_info (Dict[str, Union[float, str]]): parameters of node
            pvals (List[str]): values in parents nodes

        Returns:
            str: prediction
        """

        vals = node_info['vals']
        disct = []
        if not pvals:
            dist = node_info['cprob']
        else:
            # noinspection PyTypeChecker
            dist = node_info['cprob'][str(pvals)]
        max_value = max(dist)
        indices = [
            index for index,
            value in enumerate(dist) if value == max_value]
        max_ind = 0
        if len(indices) == 1:
            max_ind = indices[0]
        else:
            max_ind = random.choice(indices)
        return vals[max_ind]
