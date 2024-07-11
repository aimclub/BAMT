from abc import abstractmethod, ABC
from typing import Union, List, Optional, Sequence

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.objective import Objective
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements


class BaseAnalyzeApproach(ABC):
    """
    Base class for analysis approach.
    :param graph: Graph containing the analyzed Node
    :param objective: objective functions for computing metric values
    """

    def __init__(self, graph: Graph, objective: Objective,
                 requirements: StructuralAnalysisRequirements = None):
        self._graph = graph
        self._objective = objective
        self._origin_metrics = list()
        self._requirements = \
            StructuralAnalysisRequirements() if requirements is None else requirements

    @abstractmethod
    def analyze(self, entity: Union[GraphNode, Edge], **kwargs) -> Union[List[dict], List[float]]:
        """ Creates the difference metric(scorer, index, etc) of the changed
        graph in relation to the original one.
        :param entity: entity to analyze.
        """
        pass

    @abstractmethod
    def sample(self, *args) -> Union[List[Graph], Graph]:
        """ Changes the graph according to the approach. """
        pass

    @staticmethod
    def _compare_with_origin_by_metric(origin_metric: Optional[float], modified_metric: Optional[float]) -> float:
        """ Calculates one metric. """
        if not modified_metric or not origin_metric:
            return -1.0

        if modified_metric < 0.0:
            numerator = modified_metric
            denominator = origin_metric
        else:
            numerator = origin_metric
            denominator = modified_metric

        if denominator == 0:
            return -1.0

        return numerator / denominator

    def _compare_with_origin_by_metrics(self, modified_graph: Graph) -> List[float]:
        """ Returns all relative metrics calculated. """
        modified_graph_metrics = self._objective(modified_graph).values

        if not self._origin_metrics:
            self._origin_metrics = self._objective(self._graph).values

        res = []
        for i in range(len(modified_graph_metrics)):
            res.append(self._compare_with_origin_by_metric(modified_metric=modified_graph_metrics[i],
                                                           origin_metric=self._origin_metrics[i]))
        return res
