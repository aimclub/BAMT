from golem.structural_analysis.graph_sa.edge_sa_approaches import EdgeDeletionAnalyze, EdgeReplaceOperationAnalyze
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeDeletionAnalyze, NodeReplaceOperationAnalyze, \
    SubtreeDeletionAnalyze
from golem.structural_analysis.graph_sa.postproc_methods import nodes_deletion, nodes_replacement, subtree_deletion, \
    edges_deletion, edges_replacement

NODE_DELETION = 'NodeDeletionAnalyze'
NODE_REPLACEMENT = 'NodeReplaceOperationAnalyze'
SUBTREE_DELETION = 'SubtreeDeletionAnalyze'
EDGE_DELETION = 'EdgeDeletionAnalyze'
EDGE_REPLACEMENT = 'EdgeReplaceOperationAnalyze'


class StructuralAnalysisApproachesRepository:
    approaches_dict = {NODE_DELETION: {'approach': NodeDeletionAnalyze,
                                       'postproc_method': nodes_deletion},
                       NODE_REPLACEMENT: {'approach': NodeReplaceOperationAnalyze,
                                          'postproc_method': nodes_replacement},
                       SUBTREE_DELETION: {'approach': SubtreeDeletionAnalyze,
                                          'postproc_method': subtree_deletion},
                       EDGE_DELETION: {'approach': EdgeDeletionAnalyze,
                                       'postproc_method': edges_deletion},
                       EDGE_REPLACEMENT: {'approach': EdgeReplaceOperationAnalyze,
                                          'postproc_method': edges_replacement}}

    def approach_by_name(self, approach_name: str):
        return self.approaches_dict[approach_name]['approach']

    def postproc_method_by_name(self, approach_name: str):
        return self.approaches_dict[approach_name]['postproc_method']
