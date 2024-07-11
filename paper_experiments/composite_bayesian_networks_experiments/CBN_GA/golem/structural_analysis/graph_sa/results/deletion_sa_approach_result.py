from typing import List

from golem.structural_analysis.graph_sa.results.base_sa_approach_result import BaseSAApproachResult


class DeletionSAApproachResult(BaseSAApproachResult):
    """ Class for presenting deletion result approaches. """

    def __init__(self):
        self.metrics = []

    def add_results(self, metrics_values: List[float]):
        self.metrics = metrics_values

    def get_worst_result(self, metric_idx_to_optimize_by: int) -> float:
        """ Returns the worst metric among all calculated. """
        return self.metrics[metric_idx_to_optimize_by]

    def get_worst_result_with_names(self, metric_idx_to_optimize_by: int) -> dict:
        return {'value': self.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by)}

    def get_dict_results(self) -> List[float]:
        """ Returns all calculated results. """
        return self.metrics

    def get_rounded_metrics(self, idx: int = 2) -> list:
        return [round(metric, idx) for metric in self.metrics]

    def __str__(self):
        return 'DeletionSAApproachResult'
