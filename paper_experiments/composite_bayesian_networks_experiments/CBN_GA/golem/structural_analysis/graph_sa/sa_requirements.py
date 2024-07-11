import random
from collections import namedtuple
from typing import List, Optional

from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.dag.graph import GraphNode
from golem.structural_analysis.graph_sa.entities.edge import Edge

ReplacementAnalysisMetaParams = namedtuple('ReplacementAnalysisMetaParams', ['edges_to_replace_to',
                                                                             'number_of_random_operations_edges',
                                                                             'nodes_to_replace_to',
                                                                             'number_of_random_operations_nodes'])


class StructuralAnalysisRequirements:
    """
    Use this object to pass all the requirements needed for SA.

    :param graph_verifier: verifier for graph in SA.
    :param main_metric_idx: index of metric to optimize by. Other metrics will be calculated and saved if needed.
    :param replacement_nodes_to_replace_to: defines nodes which is used in replacement analysis
    :param replacement_number_of_random_operations_nodes: if replacement_nodes_to_replace_to is not filled, \
    define the number of randomly chosen operations used in replacement analysis
    :param replacement_edges_to_replace_to: defines edges which is used in replacement analysis
    :param replacement_number_of_random_operations_edges: if replacement_edges_to_replace_to is not filled, \
    define the number of randomly chosen operations used in replacement analysis
    :param is_visualize: defines whether the SA visualization needs to be saved to .png files
    :param is_save_results_to_json: defines whether the SA indices needs to be saved to .json file
    """

    def __init__(self,
                 graph_verifier: GraphVerifier = None,
                 main_metric_idx: int = 0,
                 replacement_nodes_to_replace_to: Optional[List[GraphNode]] = None,
                 replacement_number_of_random_operations_nodes: Optional[int] = 3,
                 replacement_edges_to_replace_to: Optional[List[Edge]] = None,
                 replacement_number_of_random_operations_edges: Optional[int] = 3,
                 is_visualize: bool = False,
                 is_save_results_to_json: bool = False,
                 seed: int = random.randint(0, 100)):

        self.graph_verifier = graph_verifier or GraphVerifier(DEFAULT_DAG_RULES)

        self.main_metric_idx = main_metric_idx

        self.replacement_meta = ReplacementAnalysisMetaParams(replacement_edges_to_replace_to,
                                                              replacement_number_of_random_operations_edges,
                                                              replacement_nodes_to_replace_to,
                                                              replacement_number_of_random_operations_nodes)

        self.is_visualize = is_visualize
        self.is_save = is_save_results_to_json
        self.seed = seed
