import random
from abc import ABC
from copy import deepcopy
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Type, Union, Dict, Callable

from golem.core.log import default_log
from golem.core.dag.graph import Graph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import default_data_dir
from golem.structural_analysis.base_sa_approaches import BaseAnalyzeApproach
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.results.deletion_sa_approach_result import \
    DeletionSAApproachResult
from golem.structural_analysis.graph_sa.results.object_sa_result import ObjectSAResult
from golem.structural_analysis.graph_sa.results.replace_sa_approach_result import \
    ReplaceSAApproachResult
from golem.structural_analysis.graph_sa.results.utils import EntityTypesEnum
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements, \
    ReplacementAnalysisMetaParams


class EdgeAnalysis:
    """
    Class for successively applying approaches for structural analysis

    :param approaches: methods applied to edges to modify the graph or analyze certain operations.\
    Default: [EdgeDeletionAnalyze, EdgeReplaceOperationAnalyze]
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, approaches: Optional[List[Type['EdgeAnalyzeApproach']]] = None,
                 approaches_requirements: Optional[StructuralAnalysisRequirements] = None,
                 path_to_save: Optional[str] = None):

        self.approaches = [EdgeDeletionAnalyze, EdgeReplaceOperationAnalyze] \
            if approaches is None else approaches

        self.path_to_save = \
            join(default_data_dir(), 'structural', 'edges_structural') if path_to_save is None else path_to_save
        self.log = default_log(self)

        self.approaches_requirements = \
            StructuralAnalysisRequirements() if approaches_requirements is None else approaches_requirements

    def analyze(self, graph: Graph, edge: Edge,
                objective: Callable,
                timer: Optional[OptimisationTimer] = None) -> ObjectSAResult:
        """
        Method runs Edge analysis within defined approaches

        :param graph: graph containing the analyzed Edge
        :param edge: Edge object to analyze in Graph
        :param objective: objective function for computing metric values
        :param timer: timer to check if the time allotted for structural analysis has expired
        :return: dict with Edge analysis result per approach
        """

        results = ObjectSAResult(entity_idx=
                                 f'{graph.nodes.index(edge.parent_node)}_{graph.nodes.index(edge.child_node)}',
                                 entity_type=EntityTypesEnum.edge)

        for approach in self.approaches:
            if timer is not None and timer.is_time_limit_reached():
                break

            results.add_result(approach(graph=graph,
                                        objective=objective,
                                        requirements=self.approaches_requirements,
                                        path_to_save=self.path_to_save).analyze(edge=edge))

        return results


class EdgeAnalyzeApproach(BaseAnalyzeApproach, ABC):
    """
    Base class for edge analysis approach.

    :param graph: Graph containing the analyzing Edge
    :param objective: objective function for computing metric values
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, graph: Graph, objective: Objective, requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None):
        super().__init__(graph, objective, requirements)
        self._origin_metrics = None

        self._path_to_save = \
            join(default_data_dir(), 'structural', 'edges_structural') if path_to_save is None else path_to_save
        self.log = default_log(prefix='edge_analysis')

        if not exists(self._path_to_save):
            makedirs(self._path_to_save)


class EdgeDeletionAnalyze(EdgeAnalyzeApproach):
    def __init__(self, graph: Graph, objective: Objective,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objective, requirements)

        self._path_to_save = \
            join(default_data_dir(), 'structural', 'edges_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, edge: Edge, **kwargs) -> DeletionSAApproachResult:
        """
        Receives a graph without the specified edge and tries to calculate the loss for it

        :param edge: Edge object to analyze
        :return: the ratio of modified graph score to origin score
        """
        results = DeletionSAApproachResult()
        shortened_graph = self.sample(edge)
        if shortened_graph:
            losses = self._compare_with_origin_by_metrics(shortened_graph)
            self.log.message(f'loss: {losses}')
            del shortened_graph
        else:
            self.log.warning('if remove this edge then get an invalid graph')
            losses = [-1.0] * len(self._objective.metrics)

        results.add_results(metrics_values=losses)
        return results

    def sample(self, edge: Edge) -> Optional[Graph]:
        """
        Checks if it is possible to delete an edge from the graph so that it remains valid,
        and if so, deletes

        :param edge: Edge object to delete from Graph object
        :return: Graph object without edge
        """

        graph_sample = deepcopy(self._graph)

        parent_node_index_to_delete = self._graph.nodes.index(edge.parent_node)
        parent_node_to_delete = graph_sample.nodes[parent_node_index_to_delete]

        child_node_index_to_delete = self._graph.nodes.index(edge.child_node)
        child_node_to_delete = graph_sample.nodes[child_node_index_to_delete]

        graph_sample.disconnect_nodes(parent_node_to_delete, child_node_to_delete)

        verifier = self._requirements.graph_verifier
        if not verifier.verify(graph_sample):
            self.log.warning('Can not delete edge since modified graph can not pass verification')
            return None

        return graph_sample


class EdgeReplaceOperationAnalyze(EdgeAnalyzeApproach):
    """
       Replace edge with operations available for the current task
       and evaluate the score difference
    """

    def __init__(self, graph: Graph, objective: Objective,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objective, requirements)

        self._path_to_save = \
            join(default_data_dir(), 'structural', 'edges_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, edge: Edge, **kwargs) -> ReplaceSAApproachResult:
        """
        Counts the loss on each changed graph received and returns the biggest loss and
        the graph on which it was received

        :param edge: Edge object to analyze
        :return: dictionary of the best (the biggest) loss and corresponding to it edge to replace to
        """
        result = ReplaceSAApproachResult()
        requirements: ReplacementAnalysisMetaParams = self._requirements.replacement_meta
        samples_res = self.sample(edge=edge,
                                  edges_idxs_to_replace_to=requirements.edges_to_replace_to,
                                  number_of_random_operations=requirements.number_of_random_operations_edges)

        samples = samples_res['samples']
        edges_nodes_idx_to_replace_to = samples_res['edges_nodes_idx_to_replace_to']

        loss_values = []
        for i, sample_graph in enumerate(samples):
            loss_per_sample = self._compare_with_origin_by_metrics(sample_graph)
            self.log.message(f'loss: {loss_per_sample}')
            loss_values.append(loss_per_sample)

            child_node_idx = ''
            parent_node_idx = ''
            part1, part2 = edges_nodes_idx_to_replace_to[i].__str__().split(',')
            for char in part1:
                if char.isdigit():
                    child_node_idx += char
                else:
                    continue
            for char in part2:
                if char.isdigit():
                    parent_node_idx += char
                else:
                    continue
            result.add_results(entity_to_replace_to=f'{parent_node_idx}_{child_node_idx}',
                               metrics_values=loss_per_sample)
        return result

    def sample(self, edge: Edge,
               edges_idxs_to_replace_to: Optional[List[Edge]],
               number_of_random_operations: Optional[int] = 1) \
            -> Dict[str, Union[List[Graph], List[Dict[str, int]]]]:
        """
        Tries to replace the given edge with a pool of edges available for replacement (see _edge_generation docstring)
        and validates the resulting graphs

        :param edge: Edge object to replace
        :param edges_idxs_to_replace_to: edges provided for old_edge replacement
        :param number_of_random_operations: number of replacement operations, \
        if edges_to_replace_to not provided

        :return: dictionary of sequence of Graph objects with new edges instead of old one
        and indexes of edges to which to change the given edge to get these graphs
        """

        if not edges_idxs_to_replace_to or number_of_random_operations:
            edges_idxs_to_replace_to = self._edge_generation(edge=edge,
                                                             number_of_operations=number_of_random_operations)
        samples = list()
        edges_nodes_idx_to_replace_to = list()
        for replacing_nodes_idx in edges_idxs_to_replace_to:
            sample_graph = deepcopy(self._graph)

            # disconnect nodes
            previous_parent_node_index = self._graph.nodes.index(edge.parent_node)
            previous_child_node_index = self._graph.nodes.index(edge.child_node)

            previous_parent_node = sample_graph.nodes[previous_parent_node_index]
            previous_child_node = sample_graph.nodes[previous_child_node_index]

            # connect nodes
            next_parent_node = sample_graph.nodes[replacing_nodes_idx['parent_node_idx']]
            next_child_node = sample_graph.nodes[replacing_nodes_idx['child_node_idx']]

            if next_parent_node in sample_graph.nodes and \
                    next_child_node in sample_graph.nodes:
                sample_graph.connect_nodes(next_parent_node, next_child_node)

            sample_graph.disconnect_nodes(node_parent=previous_parent_node,
                                          node_child=previous_child_node,
                                          clean_up_leftovers=False)

            verifier = self._requirements.graph_verifier
            if not verifier.verify(sample_graph):
                self.log.warning('Can not connect these nodes')
            else:
                self.log.message(f'replace edge parent: {next_parent_node}')
                self.log.message(f'replace edge child: {next_child_node}')
                samples.append(sample_graph)
                edges_nodes_idx_to_replace_to.append({'parent_node_id':
                                                     replacing_nodes_idx['parent_node_idx'],
                                                      'child_node_id':
                                                     replacing_nodes_idx['child_node_idx']})

        if not edges_nodes_idx_to_replace_to:
            res = {'samples': [self._graph], 'edges_nodes_idx_to_replace_to':
                [{'parent_node_id': self._graph.nodes.index(edge.parent_node),
                  'child_node_id': self._graph.nodes.index(edge.child_node)}]}
            return res

        return {'samples': samples, 'edges_nodes_idx_to_replace_to': edges_nodes_idx_to_replace_to}

    def _edge_generation(self, edge: Edge, number_of_operations: int = 1) -> List[Dict[str, int]]:
        """
        The method returns possible edges that can replace the given edge.
        These edges must not start at the root node, already exist in the graph and must not form cycles

        :param edge: edge to be replaced
        :param number_of_operations: limits the number of possible edges to replace to

        :return: edges with which it's possible to replace the passed edge
        """
        cur_graph = deepcopy(self._graph)

        parent_node_index = self._graph.nodes.index(edge.parent_node)
        child_node_index = self._graph.nodes.index(edge.child_node)

        parent_node = cur_graph.nodes[parent_node_index]
        child_node = cur_graph.nodes[child_node_index]

        if child_node is cur_graph.root_node and len(child_node.nodes_from) == 1:
            return []

        cur_graph.disconnect_nodes(node_parent=parent_node, node_child=child_node,
                                   clean_up_leftovers=False)

        edges_in_graph = cur_graph.get_edges()

        available_edges_idx = list()

        for parent_node in cur_graph.nodes[1:]:
            for child_node in cur_graph.nodes:
                if parent_node == child_node:
                    continue
                if [parent_node, child_node] in edges_in_graph or [child_node, parent_node] in edges_in_graph:
                    continue
                if cur_graph.nodes.index(parent_node) == child_node_index and \
                        cur_graph.nodes.index(child_node) == parent_node_index:
                    continue
                available_edges_idx.append({'parent_node_idx': cur_graph.nodes.index(parent_node),
                                            'child_node_idx': cur_graph.nodes.index(child_node)})

        edges_for_replacement = random.sample(available_edges_idx, min(number_of_operations, len(available_edges_idx)))
        return edges_for_replacement
