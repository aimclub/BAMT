from typing import List, Dict

from golem.structural_analysis.graph_sa.results.base_sa_approach_result import BaseSAApproachResult


class ReplaceSAApproachResult(BaseSAApproachResult):
    """ Class for presenting replacing result approaches. """
    def __init__(self):
        """ Main dictionary `self.metrics` contains entities as key and
        list with metrics as values"""
        self.metrics = dict()

    def add_results(self, entity_to_replace_to: str, metrics_values: List[float]):
        """ Sets value for specified metric. """
        self.metrics[entity_to_replace_to] = metrics_values

    def get_worst_result(self, metric_idx_to_optimize_by: int) -> float:
        """ Returns value of the worst metric. """
        return max([metrics[metric_idx_to_optimize_by] for metrics in list(self.metrics.values())])

    def get_worst_result_with_names(self, metric_idx_to_optimize_by: int) -> dict:
        """ Returns the worst metric among all calculated with its name and node's to replace to name. """
        worst_value = self.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by)
        for entity in self.metrics:
            if list(self.metrics[entity])[metric_idx_to_optimize_by] == worst_value:
                return {'value': worst_value, 'entity_to_replace_to': entity}

    def get_dict_results(self) -> Dict[int, List[float]]:
        """ Returns dict representation of results. """
        return self.metrics

    def get_rounded_metrics(self, idx: int = 2) -> dict:
        rounded = {}
        for metric in self.metrics:
            rounded[metric] = [round(metric, idx) for metric in self.metrics[metric]]
        return rounded

    def __str__(self):
        return 'ReplaceSAApproachResult'
