import random
from abc import ABC
from copy import deepcopy
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Type, Union, Any

from golem.core.log import default_log
from golem.core.dag.graph import Graph, GraphNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import default_data_dir
from golem.structural_analysis.base_sa_approaches import BaseAnalyzeApproach
from golem.structural_analysis.graph_sa.results.deletion_sa_approach_result import \
    DeletionSAApproachResult
from golem.structural_analysis.graph_sa.results.object_sa_result import ObjectSAResult
from golem.structural_analysis.graph_sa.results.replace_sa_approach_result import \
    ReplaceSAApproachResult
from golem.structural_analysis.graph_sa.results.utils import EntityTypesEnum
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements, \
    ReplacementAnalysisMetaParams


class NodeAnalysis:
    """
    :param approaches: methods applied to nodes to modify the graph or analyze certain operations.\
    Default: [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceOperationAnalyze]
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, node_factory: Any,
                 approaches: Optional[List[Type['NodeAnalyzeApproach']]] = None,
                 approaches_requirements: Optional[StructuralAnalysisRequirements] = None,
                 path_to_save: Optional[str] = None):

        self.node_factory = node_factory

        self.approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze] if approaches is None else approaches

        self.path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        self.log = default_log(self)

        self.approaches_requirements = \
            StructuralAnalysisRequirements() if approaches_requirements is None else approaches_requirements

    def analyze(self, graph: Graph, node: GraphNode,
                objective: Objective,
                timer: Optional[OptimisationTimer] = None) -> ObjectSAResult:

        """
        Method runs Node analysis within defined approaches

        :param graph: Graph containing the analyzed Node
        :param node: Node object to analyze in Graph
        :param objective: objective function for computing metric values
        :param timer: timer to check if the time allotted for structural analysis has expired
        :return: dict with Node analysis result per approach
        """

        results = ObjectSAResult(entity_idx=str(graph.nodes.index(node)),
                                 entity_type=EntityTypesEnum.node)

        for approach in self.approaches:
            if timer is not None and timer.is_time_limit_reached():
                break

            results.add_result(approach(graph=graph,
                                        objective=objective,
                                        node_factory=self.node_factory,
                                        requirements=self.approaches_requirements,
                                        path_to_save=self.path_to_save).analyze(node=node))
        return results


class NodeAnalyzeApproach(BaseAnalyzeApproach, ABC):
    """
    Base class for node analysis approach.
    :param graph: Graph containing the analyzed Node
    :param objective: objective functions for computing metric values
    :param node_factory: node factory to choose nodes to replace to
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, graph: Graph, objective: Objective, node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objective, requirements)
        self._node_factory = node_factory

        self._origin_metrics = list()
        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        self.log = default_log(prefix='node_analysis')

        if not exists(self._path_to_save):
            makedirs(self._path_to_save)


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, graph: Graph, objective: Objective,
                 node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objective, node_factory, requirements)
        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: GraphNode, **kwargs) -> DeletionSAApproachResult:
        """
        Receives a graph without the specified node and tries to calculate the loss for it

        :param node: GraphNode object to analyze
        :return: the ratio of modified graph score to origin score
        """
        results = DeletionSAApproachResult()
        if node is self._graph.root_node:
            self.log.warning(f'{node} node can not be deleted')
            results.add_results(metrics_values=[-1.0] * len(self._objective.metrics))
            return results
        else:
            shortened_graph = self.sample(node)
            if shortened_graph:
                losses = self._compare_with_origin_by_metrics(shortened_graph)
                self.log.message(f'losses for {node.name}: {losses}')
                del shortened_graph
            else:
                losses = [-1.0] * len(self._objective.metrics)

            results.add_results(metrics_values=losses)
            return results

    def sample(self, node: GraphNode):
        """
        Checks if it is possible to delete the node from the graph so that it remains valid,
        and if so, deletes

        :param node: GraphNode object to delete from Graph object
        :return: Graph object without node
        """
        graph_sample = deepcopy(self._graph)
        node_index_to_delete = self._graph.nodes.index(node)
        node_to_delete = graph_sample.nodes[node_index_to_delete]

        if node_to_delete.name == 'class_decompose':
            for child in graph_sample.node_children(node_to_delete):
                graph_sample.delete_node(child)

        graph_sample.delete_node(node_to_delete)

        verifier = self._requirements.graph_verifier
        if not verifier.verify(graph_sample):
            self.log.message('Can not delete node since modified graph can not be verified')
            return None

        return graph_sample


class NodeReplaceOperationAnalyze(NodeAnalyzeApproach):
    """
    Replace node with operations available for the current task
    and evaluate the score difference
    """

    def __init__(self, graph: Graph, objective: Objective,
                 node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objective, node_factory, requirements)

        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: GraphNode, **kwargs) -> ReplaceSAApproachResult:
        """
        Counts the loss on each changed graph received and returns losses

        :param node: GraphNode object to analyze

        :return: the ratio of modified graph score to origin score
        """
        result = ReplaceSAApproachResult()
        requirements: ReplacementAnalysisMetaParams = self._requirements.replacement_meta
        node_id = self._graph.nodes.index(node)
        samples = self.sample(node=node,
                              nodes_to_replace_to=requirements.nodes_to_replace_to,
                              number_of_random_operations=requirements.number_of_random_operations_nodes)

        for sample_graph in samples:
            loss_per_sample = self._compare_with_origin_by_metrics(sample_graph)
            self.log.message(f'losses: {loss_per_sample}\n')

            result.add_results(entity_to_replace_to=sample_graph.nodes[node_id].name, metrics_values=loss_per_sample)

        return result

    def sample(self, node: GraphNode,
               nodes_to_replace_to: Optional[List[GraphNode]],
               number_of_random_operations: int = 1) -> Union[List[Graph], Graph]:
        """
        Replaces the given node with a pool of nodes available for replacement (see _node_generation docstring)

        :param node: GraphNode object to replace
        :param nodes_to_replace_to: nodes provided for old_node replacement
        :param number_of_random_operations: number of replacement operations, \
        if nodes_to_replace_to not provided
        :return: Sequence of Graph objects with new operations instead of old one
        """

        if not nodes_to_replace_to:
            nodes_to_replace_to = self._node_generation(node=node,
                                                        node_factory=self._node_factory,
                                                        number_of_operations=number_of_random_operations)

        samples = list()
        for replacing_node in nodes_to_replace_to:
            sample_graph = deepcopy(self._graph)
            replaced_node_index = self._graph.nodes.index(node)
            replaced_node = sample_graph.nodes[replaced_node_index]
            sample_graph.update_node(old_node=replaced_node,
                                     new_node=replacing_node)
            verifier = self._requirements.graph_verifier
            if not verifier.verify(sample_graph):
                self.log.warning(f'Can not replace {node.name} node with {replacing_node.name} node.')
            else:
                self.log.message(f'replacing node: {replacing_node.name}')
                samples.append(sample_graph)

        if not samples:
            samples.append(self._graph)

        return samples

    @staticmethod
    def _node_generation(node: GraphNode,
                         node_factory: OptNodeFactory,
                         number_of_operations: int = 1) -> List[GraphNode]:
        """
        The method returns possible nodes that can replace the given node

        :param node: the node to be replaced
        :param number_of_operations: limits the number of possible nodes to replace to

        :return: nodes that can be used to replace
        """

        available_nodes = []
        for i in range(number_of_operations):
            available_nodes.append(node_factory.exchange_node(node=node))

        if number_of_operations:
            available_nodes = [i for i in available_nodes if i != node.name]
            number_of_operations = min(len(available_nodes), number_of_operations)
            random_nodes = random.sample(available_nodes, number_of_operations)
        else:
            random_nodes = available_nodes

        return random_nodes


class SubtreeDeletionAnalyze(NodeAnalyzeApproach):
    """
    Approach to delete specified node subtree
    """

    def __init__(self, graph: Graph, objective: Objective,
                 node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objective, node_factory, requirements)
        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural')\
            if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: GraphNode, **kwargs) -> DeletionSAApproachResult:
        """
        Receives a graph without the specified node's subtree and
        tries to calculate the loss for it

        :param node: GraphNode object to analyze
        :return: the ratio of modified graph score to origin score
        """
        results = DeletionSAApproachResult()
        if node is self._graph.root_node:
            self.log.warning(f'{node} subtree can not be deleted')
            results.add_results(metrics_values=[-1.0] * len(self._objective.metrics))
            return results
        else:
            shortened_graph = self.sample(node)
            if shortened_graph:
                losses = self._compare_with_origin_by_metrics(shortened_graph)
                self.log.message(f'losses for {node.name}: {losses}')
                del shortened_graph
            else:
                losses = [-1.0] * len(self._objective.metrics)

            results.add_results(metrics_values=losses)
            return results

    def sample(self, node: GraphNode):
        """
        Checks if it is possible to delete the node's subtree from the graph so that it remains valid,
        and if so, deletes

        :param node: GraphNode object from which to delete subtree from Graph object
        :return: Graph object without subtree
        """
        graph_sample = deepcopy(self._graph)
        node_index_to_delete = self._graph.nodes.index(node)
        node_to_delete = graph_sample.nodes[node_index_to_delete]

        if node_to_delete.name == 'class_decompose':
            for child in graph_sample.node_children(node_to_delete):
                graph_sample.delete_node(child)

        graph_sample.delete_subtree(node_to_delete)

        verifier = self._requirements.graph_verifier
        if not verifier.verify(graph_sample):
            self.log.warning('Can not delete subtree since modified graph can not pass verification')
            return None

        return graph_sample
