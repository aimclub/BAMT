from typing import List

from golem.structural_analysis.graph_sa.results.base_sa_approach_result import BaseSAApproachResult
from golem.structural_analysis.graph_sa.results.deletion_sa_approach_result import \
    DeletionSAApproachResult
from golem.structural_analysis.graph_sa.results.replace_sa_approach_result import \
    ReplaceSAApproachResult
from golem.structural_analysis.graph_sa.results.utils import EntityTypesEnum

NODE_DELETION = 'NodeDeletionAnalyze'
NODE_REPLACEMENT = 'NodeReplaceOperationAnalyze'
SUBTREE_DELETION = 'SubtreeDeletionAnalyze'
EDGE_DELETION = 'EdgeDeletionAnalyze'
EDGE_REPLACEMENT = 'EdgeReplaceOperationAnalyze'


class StructuralAnalysisResultsRepository:
    approaches_dict = {NODE_DELETION: {'result_class': DeletionSAApproachResult},
                       NODE_REPLACEMENT: {'result_class': ReplaceSAApproachResult},
                       SUBTREE_DELETION: {'result_class': DeletionSAApproachResult},
                       EDGE_DELETION: {'result_class': DeletionSAApproachResult},
                       EDGE_REPLACEMENT: {'result_class': ReplaceSAApproachResult}}

    def get_method_by_result_class(self, result_class: BaseSAApproachResult, entity_class: str) -> str:
        for method in self.approaches_dict.keys():
            if self.approaches_dict[method]['result_class'] == result_class.__class__ \
                    and entity_class in method.lower():
                return method

    def get_class_by_str(self, result_str: str) -> BaseSAApproachResult:
        for method in self.approaches_dict.keys():
            if result_str == method:
                return self.approaches_dict[method]['result_class']


class ObjectSAResult:
    """ Class specifying results of Structural Analysis for one entity(node or edge). """

    def __init__(self, entity_idx: str, entity_type: EntityTypesEnum):
        self.entity_idx = entity_idx
        self.entity_type = entity_type
        self.result_approaches: List[BaseSAApproachResult] = []

    def get_worst_result(self, metric_idx_to_optimize_by: int) -> float:
        """ Returns the worst result among all result classes. """
        worst_results = []
        for approach in self.result_approaches:
            worst_results.append(approach.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by))
        if not worst_results:
            return 0
        return max(worst_results)

    def get_worst_result_with_names(self, metric_idx_to_optimize_by: int) -> dict:
        """ Returns worst result with additional information. """
        worst_result = self.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by)
        for approach in self.result_approaches:
            if approach.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by) == worst_result:
                sa_approach_name = self._get_approach_name(approach=approach)
                result = {'entity_idx': self.entity_idx, 'approach_name': sa_approach_name}
                result.update(approach.get_worst_result_with_names(metric_idx_to_optimize_by=metric_idx_to_optimize_by))
                return result

    def add_result(self, result: BaseSAApproachResult):
        self.result_approaches.append(result)

    def get_dict_results(self) -> dict:
        """ Returns dict representation of results. """
        results = dict()
        for approach in self.result_approaches:
            sa_approach_name = self._get_approach_name(approach=approach)
            results[sa_approach_name] = approach.get_dict_results()
        return {self.entity_idx: results}

    def _get_approach_name(self, approach: BaseSAApproachResult) -> str:
        sa_approach_name = StructuralAnalysisResultsRepository() \
            .get_method_by_result_class(approach, self.entity_type.value)
        return sa_approach_name
