import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import multiprocessing

from golem.core.log import default_log
from golem.core.dag.graph import Graph, GraphNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.timer import OptimisationTimer
from golem.structural_analysis.graph_sa.edge_sa_approaches import EdgeAnalyzeApproach, EdgeDeletionAnalyze, \
    EdgeReplaceOperationAnalyze
from golem.structural_analysis.graph_sa.edges_analysis import EdgesAnalysis
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeAnalyzeApproach, NodeDeletionAnalyze, \
    NodeReplaceOperationAnalyze, SubtreeDeletionAnalyze
from golem.structural_analysis.graph_sa.nodes_analysis import NodesAnalysis
from golem.structural_analysis.graph_sa.results.object_sa_result import ObjectSAResult
from golem.structural_analysis.graph_sa.results.sa_analysis_results import SAAnalysisResults
from golem.structural_analysis.graph_sa.results.utils import EntityTypesEnum
from golem.structural_analysis.graph_sa.sa_approaches_repository import StructuralAnalysisApproachesRepository
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements
from golem.visualisation.graph_viz import NodeColorType


class GraphStructuralAnalysis:
    """
    This class works as facade and allows to apply all kind of approaches
    to whole graph and separate nodes together.

    :param objective: list of objective functions for computing metric values
    :param node_factory: node factory to advise changes from available operations and models
    :param approaches: methods applied to graph. Default: None
    :param requirements: extra requirements to define specific details for different approaches.\
    See StructuralAnalysisRequirements class documentation.
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural/
    Default: False
    """

    def __init__(self, objective: Objective,
                 node_factory: OptNodeFactory,
                 is_preproc: bool = True,
                 approaches: List = None,
                 requirements: StructuralAnalysisRequirements = StructuralAnalysisRequirements(),
                 path_to_save: str = None,
                 is_visualize_per_iteration: bool = False):

        self.is_preproc = is_preproc
        self._log = default_log(self)

        if approaches:
            self.nodes_analyze_approaches = [approach for approach in approaches
                                             if issubclass(approach, NodeAnalyzeApproach)]
            self.edges_analyze_approaches = [approach for approach in approaches
                                             if issubclass(approach, EdgeAnalyzeApproach)]
        else:
            self._log.message('Approaches for analysis are not given, thus will be set to defaults.')
            self.nodes_analyze_approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze,
                                             SubtreeDeletionAnalyze]
            self.edges_analyze_approaches = [EdgeDeletionAnalyze, EdgeReplaceOperationAnalyze]

        self._nodes_analyze = NodesAnalysis(objective=objective,
                                            node_factory=node_factory,
                                            approaches=self.nodes_analyze_approaches,
                                            requirements=requirements,
                                            path_to_save=path_to_save)

        self._edges_analyze = EdgesAnalysis(objective=objective,
                                            approaches=self.edges_analyze_approaches,
                                            requirements=requirements,
                                            path_to_save=path_to_save)

        self.main_metric_idx = requirements.main_metric_idx
        self.path_to_save = path_to_save
        self.is_visualize_per_iteration = is_visualize_per_iteration

    def analyze(self, graph: Graph,
                result: SAAnalysisResults = None,
                nodes_to_analyze: List[GraphNode] = None, edges_to_analyze: List[Edge] = None,
                n_jobs: int = 1, timer: OptimisationTimer = None) -> SAAnalysisResults:
        """
        Applies defined structural analysis approaches

        :param graph: graph object to analyze
        :param result: analysis result
        :param nodes_to_analyze: nodes to analyze. Default: all nodes
        :param edges_to_analyze: edges to analyze. Default: all edges
        :param n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's).
        Tip: if specified graph isn't huge (as NN, for example) than set n_jobs to default value.
        :param timer: timer with timeout left for optimization
        """

        if not result:
            result = SAAnalysisResults()

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if self.is_preproc:
            graph = self.graph_preprocessing(graph=graph)

        if self.nodes_analyze_approaches:
            self._nodes_analyze.analyze(graph=graph,
                                        results=result,
                                        nodes_to_analyze=nodes_to_analyze,
                                        n_jobs=n_jobs, timer=timer)

        if self.edges_analyze_approaches:
            self._edges_analyze.analyze(graph=graph,
                                        results=result,
                                        edges_to_analyze=edges_to_analyze,
                                        n_jobs=n_jobs, timer=timer)

        return result

    def optimize(self, graph: Graph,
                 n_jobs: int = 1, timer: OptimisationTimer = None,
                 max_iter: int = 10) -> Tuple[Graph, SAAnalysisResults]:
        """ Optimizes graph by applying 'analyze' method and deleting/replacing parts
        of graph iteratively
        :param graph: graph object to analyze.
        :param n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's).
        Tip: if specified graph isn't huge (as NN, for example) than set n_jobs to default value.
        :param timer: timer with timeout left for optimization.
        :param max_iter: max number of iterations of analysis. """

        approaches_repo = StructuralAnalysisApproachesRepository()
        approaches = self._nodes_analyze.approaches + self._edges_analyze.approaches
        approaches_names = [approach.__name__ for approach in approaches]

        # what actions were applied on the graph and how many
        actions_applied = dict.fromkeys(approaches_names, 0)

        result = SAAnalysisResults()

        analysis_result = self.analyze(graph=graph, result=result, timer=timer, n_jobs=n_jobs)
        converged = False
        iter = 0

        if analysis_result.is_empty:
            self._log.message(f'{iter} actions were taken during SA')
            return graph, analysis_result

        while not converged:
            iter += 1
            worst_result = analysis_result.get_info_about_worst_result(
                metric_idx_to_optimize_by=self.main_metric_idx)
            if self.is_visualize_per_iteration:
                self.visualize_on_graph(graph=deepcopy(graph), analysis_result=analysis_result,
                                        metric_idx_to_optimize_by=self.main_metric_idx,
                                        mode='final',
                                        font_size_scale=0.6)
            if worst_result['value'] > 1:
                # apply the worst approach
                postproc_method = approaches_repo.postproc_method_by_name(worst_result['approach_name'])
                graph = postproc_method(graph=graph, worst_result=worst_result)
                actions_applied[f'{worst_result["approach_name"]}'] += 1

                if timer is not None and timer.is_time_limit_reached():
                    break

                if max_iter and iter >= max_iter:
                    break

                analysis_result = self.analyze(graph=graph,
                                               result=result,
                                               n_jobs=n_jobs,
                                               timer=timer)
            else:
                converged = True

        self._log.message(f'{iter} iterations passed during SA')
        self._log.message(f'The following actions were applied during SA: {actions_applied}')

        if self.path_to_save:
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
            analysis_result.save(path=self.path_to_save)

        return graph, analysis_result

    @staticmethod
    def apply_results(graph: Graph, analysis_result: SAAnalysisResults,
                      metric_idx_to_optimize_by: int, iter: int = None) -> Graph:
        """ Optimizes graph by applying actions specified in analysis_result. """

        def optimize_on_iter(graph: Graph, analysis_result: SAAnalysisResults,
                             metric_idx_to_optimize_by: int, iter: int = None) -> Graph:
            """ Get worst result on specified iteration and process graph with it. """
            worst_result = analysis_result.get_info_about_worst_result(
                metric_idx_to_optimize_by=metric_idx_to_optimize_by, iter=iter)
            approaches_repo = StructuralAnalysisApproachesRepository()
            postproc_method = approaches_repo.postproc_method_by_name(worst_result['approach_name'])
            graph = postproc_method(graph=graph, worst_result=worst_result)
            return graph

        if iter is not None:
            return optimize_on_iter(graph=graph, analysis_result=analysis_result,
                                    metric_idx_to_optimize_by=metric_idx_to_optimize_by, iter=iter)

        num_of_iter = len(analysis_result.results_per_iteration)
        for i in range(num_of_iter):
            graph = optimize_on_iter(graph=graph, analysis_result=analysis_result,
                                     metric_idx_to_optimize_by=metric_idx_to_optimize_by, iter=iter)
        return graph

    @staticmethod
    def visualize_on_graph(graph: Graph, analysis_result: SAAnalysisResults,
                           metric_idx_to_optimize_by: int, mode: str = 'final',
                           save_path: str = None, node_color: Optional[NodeColorType] = None, dpi: Optional[int] = None,
                           node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
                           edge_curvature_scale: Optional[float] = None):
        """ Visualizes results of Structural Analysis on graph(s).
        :param graph: initial graph before SA
        :param analysis_result: results of Structural Analysis
        :param metric_idx_to_optimize_by: index of optimized metric
        :param mode: 'first' -- visualize only first iteration of SA,
                     'final' - visualize only the last iteration of SA,
                     'by_iteration' -- visualize every iteration of SA.
        :param save_path: path to save visualizations
        :param node_color: color of nodes to use.
        :param node_size_scale: use to make node size bigger or lesser. Supported only for the engine 'matplotlib'.
        :param font_size_scale: use to make font size bigger or lesser. Supported only for the engine 'matplotlib'.
        :param edge_curvature_scale: use to make edges more or less curved. Supported only for the engine 'matplotlib'.
        :param dpi: DPI of the output image. Not supported for the engine 'pyvis'.
        """

        def get_nodes_and_edges_labels(analysis_result: SAAnalysisResults, iter: int) \
                -> Tuple[Dict[int, str], Dict[int, str]]:
            """ Get nodes and edges labels in dictionary form. """

            def get_str_labels(result: ObjectSAResult) -> str:
                """ Get string results. """
                approaches = result.result_approaches
                cur_label = ''
                for approach in approaches:
                    approach_name = result._get_approach_name(approach=approach)
                    if 'del' in approach_name.lower():
                        short_approach_name = 'D'
                    else:
                        short_approach_name = 'R'
                    cur_label += f'{short_approach_name}: {approach.get_rounded_metrics(idx=2)}\n'
                return cur_label

            nodes_labels = {}
            for i, node_result in enumerate(analysis_result.results_per_iteration[iter][EntityTypesEnum.node.value]):
                nodes_labels[i] = get_str_labels(result=node_result)

            edges_labels = {}
            for i, edge_result in enumerate(analysis_result.results_per_iteration[iter][EntityTypesEnum.edge.value]):
                edges_labels[i] = get_str_labels(result=edge_result)

            return nodes_labels, edges_labels

        num_of_iter = len(analysis_result.results_per_iteration)

        if mode == 'first':
            iters = [0]
        else:
            iters = range(num_of_iter)

        for i in iters:
            nodes_labels, edges_labels = get_nodes_and_edges_labels(analysis_result=analysis_result, iter=i)
            if not (mode == 'final' and i != iters[-1]):
                if not Path.is_file(Path(save_path)):
                    j = 0
                    while f'pipeline_after_sa_{j}.png' in os.listdir(save_path):
                        j += 1
                    cur_path = os.path.join(save_path, f'pipeline_after_sa_{j}.png')
                graph.show(node_color=node_color, dpi=dpi, node_size_scale=node_size_scale,
                           nodes_labels=nodes_labels, font_size_scale=font_size_scale,
                           edge_curvature_scale=edge_curvature_scale,
                           edges_labels=edges_labels, save_path=cur_path)
                default_log("SA_visualization").info(f"SA visualization was saved to: {cur_path}")
            if mode == 'by_iteration':
                graph = GraphStructuralAnalysis.apply_results(graph=graph, analysis_result=analysis_result,
                                                              metric_idx_to_optimize_by=metric_idx_to_optimize_by,
                                                              iter=i)

    @staticmethod
    def graph_preprocessing(graph: Graph) -> Graph:
        """ Graph preprocessing, which consists in removing consecutive nodes
        with the same models/operations in the graph """
        for node_child in reversed(graph.nodes):
            if not node_child.nodes_from or len(node_child.nodes_from) != 1:
                continue
            nodes_uid_to_delete = []
            for node_parent in node_child.nodes_from:
                if node_child.name == node_parent.name:
                    nodes_uid_to_delete.append(node_parent.uid)
            # there is a need to store nodes using uid since after deleting one of the nodes in graph
            # other nodes will not remain the same (nodes_from may be changed)
            for uid in nodes_uid_to_delete:
                node_to_delete = [node for node in graph.nodes if node.uid == uid][0]
                graph.delete_node(node_to_delete)
        return graph
