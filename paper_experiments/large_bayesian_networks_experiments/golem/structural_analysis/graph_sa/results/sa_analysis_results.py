import json
import os.path
from datetime import datetime
from typing import List, Optional, Union

from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.paths import project_root
from golem.utilities.serializable import Serializable
from golem.serializers import Serializer
from golem.structural_analysis.graph_sa.results.deletion_sa_approach_result import DeletionSAApproachResult
from golem.structural_analysis.graph_sa.results.object_sa_result import ObjectSAResult, \
    StructuralAnalysisResultsRepository
from golem.structural_analysis.graph_sa.results.utils import EntityTypesEnum


class SAAnalysisResults(Serializable):
    """ Class presenting results of Structural Analysis for the whole graph. """

    def __init__(self):
        self.results_per_iteration = {}
        self._add_empty_iteration_results()
        self.log = default_log('sa_results')

    def _add_empty_iteration_results(self):
        last_iter_num = int(list(self.results_per_iteration.keys())[-1]) if self.results_per_iteration.keys() else -1
        self.results_per_iteration.update({(last_iter_num + 1): self._init_iteration_result()})

    @staticmethod
    def _init_iteration_result() -> dict:
        return {EntityTypesEnum.node.value: [], EntityTypesEnum.edge.value: []}

    @property
    def is_empty(self) -> bool:
        """ Bool value indicating is there any calculated results. """
        if self.results_per_iteration[0] is None and \
                self.results_per_iteration[0] is None:
            return True
        return False

    def get_info_about_worst_result(self, metric_idx_to_optimize_by: int, iter: Optional[int] = None) -> dict:
        """ Returns info about the worst result.
        :param metric_idx_to_optimize_by: metric idx to optimize by
        :param iter: iteration on which to search for. """
        worst_value = None
        worst_result = None
        if iter is None:
            iter = list(self.results_per_iteration.keys())[-1]

        nodes_results = self.results_per_iteration[iter][EntityTypesEnum.node.value]
        edges_results = self.results_per_iteration[iter][EntityTypesEnum.edge.value]

        for i, res in enumerate(nodes_results + edges_results):
            cur_res = res.get_worst_result_with_names(
                metric_idx_to_optimize_by=metric_idx_to_optimize_by)
            if not worst_value or cur_res['value'] > worst_value:
                worst_value = cur_res['value']
                worst_result = cur_res
        return worst_result

    def add_results(self, results: List[ObjectSAResult]):
        if not results:
            return
        key = results[0].entity_type.value
        iter_num = self._get_last_empty_iter(key=key)
        for result in results:
            self.results_per_iteration[iter_num][key].append(result)

    def _get_last_empty_iter(self, key: str) -> int:
        """ Returns number of last iteration with empty key field. """
        for i, result in enumerate(self.results_per_iteration.values()):
            if not result[key]:
                return i
        self._add_empty_iteration_results()
        return list(self.results_per_iteration.keys())[-1]

    def save(self, path: str = None, datetime_in_path: bool = True) -> dict:
        """ Saves SA results in json format. """
        dict_results = dict()
        for iter in self.results_per_iteration.keys():
            dict_results[iter] = {}
            iter_result = self.results_per_iteration[iter]
            for entity_type in iter_result.keys():
                if entity_type not in dict_results[iter].keys():
                    dict_results[iter][entity_type] = {}
                for entity in iter_result[entity_type]:
                    dict_results[iter][entity_type].update(entity.get_dict_results())

        json_data = json.dumps(dict_results, cls=Serializer)

        if not path:
            path = os.path.join(project_root(), 'sa', 'sa_results.json')
        if datetime_in_path:
            file_name = os.path.basename(path).split('.')[0]
            file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{file_name}.json"
            path = os.path.join(os.path.dirname(path), 'sa')
            if not os.path.exists(path):
                os.makedirs(path)
            path = os.path.join(path, file_name)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_data)
            self.log.debug(f'SA results saved in the path: {path}.')

        return dict_results

    @staticmethod
    def load(source: Union[str, dict], graph: Optional[Graph] = None) -> 'SAAnalysisResults':
        """ Loads SA results from json format. """
        if isinstance(source, str):
            source = json.load(open(source))

        sa_result = SAAnalysisResults()
        results_repo = StructuralAnalysisResultsRepository()

        for iter in source:
            for entity_type in source[iter]:
                type_list = []
                for entity_idx in source[iter][entity_type]:
                    cur_result = ObjectSAResult(entity_idx=entity_idx,
                                                entity_type=EntityTypesEnum(entity_type))
                    dict_results = source[iter][entity_type][entity_idx]
                    for approach in dict_results:
                        app = results_repo.get_class_by_str(approach)()
                        if isinstance(app, DeletionSAApproachResult):
                            app.add_results(metrics_values=dict_results[approach])
                        else:
                            for entity_to_replace_to in dict_results[approach]:
                                app.add_results(entity_to_replace_to=entity_to_replace_to,
                                                metrics_values=dict_results[approach][entity_to_replace_to])
                        cur_result.add_result(app)
                    type_list.append(cur_result)
                sa_result.add_results(type_list)

        return sa_result
