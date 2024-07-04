from os.path import join
from typing import Optional, List, Type
import multiprocessing

from golem.core.log import default_log
from golem.core.dag.graph import Graph, GraphNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import default_data_dir
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeAnalyzeApproach, NodeAnalysis
from golem.structural_analysis.graph_sa.results.sa_analysis_results import SAAnalysisResults
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements


class NodesAnalysis:
    """
    This class is for nodes structural analysis within a Graph .
    It takes nodes and approaches to be applied to chosen nodes.
    To define which nodes to analyze pass them to nodes_to_analyze filed
    or all nodes will be analyzed.

    :param objective: objective functions for computing metric values
    :param node_factory: node factory to advise changes from available operations and models
    :param approaches: methods applied to nodes to modify the graph or analyze certain operations.\
    Default: [NodeDeletionAnalyze, NodeReplaceOperationAnalyze]
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, objective: Objective,
                 node_factory: OptNodeFactory,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None,
                 requirements: Optional[StructuralAnalysisRequirements] = None,
                 path_to_save: Optional[str] = None):

        self.objective = objective
        self.node_factory = node_factory
        self.approaches = approaches
        self.requirements = \
            StructuralAnalysisRequirements() if requirements is None else requirements
        self.log = default_log(self)
        self.path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save

    def analyze(self, graph: Graph, results: SAAnalysisResults = None,
                nodes_to_analyze: Optional[List[GraphNode]] = None,
                n_jobs: int = 1, timer: Optional[OptimisationTimer] = None) -> SAAnalysisResults:
        """
        Main method to run the analyze process for every node.

        :param graph: graph object to analyze
        :param results: SA results
        :param nodes_to_analyze: nodes to analyze. Default: all nodes
        :param n_jobs: n_jobs
        :param timer: timer indicating how much time is left for optimization
        :return nodes_results: dict with analysis result per GraphNode
        """

        if not results:
            results = SAAnalysisResults()

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if not nodes_to_analyze:
            self.log.message('Nodes to analyze are not defined. All nodes will be analyzed.')
            nodes_to_analyze = graph.nodes

        node_analysis = NodeAnalysis(approaches=self.approaches,
                                     approaches_requirements=self.requirements,
                                     node_factory=self.node_factory,
                                     path_to_save=self.path_to_save)

        with multiprocessing.Pool(processes=n_jobs) as pool:
            cur_nodes_results = pool.starmap(node_analysis.analyze,
                                       [[graph, node, self.objective, timer]
                                        for node in nodes_to_analyze])

        results.add_results(cur_nodes_results)

        return results
