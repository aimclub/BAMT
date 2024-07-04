import datetime
from collections import UserDict

from typing import Dict, Any

from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.log import LoggerAdapter, default_log
from golem.core.optimisers.dynamic_graph_requirements import DynamicGraphRequirements
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.utilities.utilities import determine_n_jobs


class ApiParams(UserDict):
    """
    Class to further distribute params specified params in API between the following classes:
    `GraphRequirements`, `GraphGenerationParams`, `GPAlgorithmParameters`.
    """
    def __init__(self, input_params: Dict[str, Any], n_jobs: int = -1, timeout: float = 5):
        self.log: LoggerAdapter = default_log(self)
        self.n_jobs: int = determine_n_jobs(n_jobs)
        self.timeout = timeout

        self._input_params = input_params
        self._input_params['timeout'] = timeout if isinstance(timeout, datetime.timedelta) else datetime.timedelta(minutes=timeout)
        self._default_common_params = self.get_default_common_params()
        super().__init__(self._input_params)

    def get_default_common_params(self):
        """ Common params that do not belong to any category
        (from `GPAlgorithmParameters`, `GraphGenerationParams`, `GraphRequirements`). """
        default_common_params = {
            'optimizer': EvoGraphOptimizer,
            'initial_graphs': list(),
            'objective': None
        }
        self.log.info("EvoGraphOptimizer was used as default optimizer, "
                      "will be overwritten by specified one if there is any.")
        return default_common_params

    def get_default_graph_generation_params(self):
        """ Default graph generations params to minimize the number of arguments that must be specified in API.
        Need to be hardcoded like that since the list of input arguments is not the same as the class fields list. """
        default_graph_generation_params = {
            'adapter': BaseNetworkxAdapter(),
            'rules_for_constraint': DEFAULT_DAG_RULES,
            'advisor': None,
            'node_factory': None,
            'random_graph_factory': None,
            'available_node_types': None,
            'remote_evaluator': None
        }
        self.log.info("BaseNetworkxAdapter was used as default adapter, "
                      "will be overwritten by specified one if there is any.")
        return default_graph_generation_params

    def get_gp_algorithm_parameters(self) -> GPAlgorithmParameters:
        default_gp_algorithm_params_dict = dict(list(vars(GPAlgorithmParameters()).items()))
        k_pop = []
        for k, v in self._input_params.items():
            if k in default_gp_algorithm_params_dict:
                default_gp_algorithm_params_dict[k] = self._input_params[k]
                k_pop.append(k)
        for k in k_pop:
            self._input_params.pop(k)
        return GPAlgorithmParameters(**default_gp_algorithm_params_dict)

    def get_graph_generation_parameters(self) -> GraphGenerationParams:
        default_graph_generation_params_dict = self.get_default_graph_generation_params()
        k_pop = []
        for k, v in self._input_params.items():
            if k in default_graph_generation_params_dict:
                default_graph_generation_params_dict[k] = self._input_params[k]
                k_pop.append(k)
        for k in k_pop:
            self._input_params.pop(k)
        ggp = GraphGenerationParams(**default_graph_generation_params_dict)
        return ggp

    def get_graph_requirements(self) -> GraphRequirements:
        default_graph_requirements_params_dict = dict(list(vars(GraphRequirements()).items()))
        # if there are any custom domain specific graph requirements params
        is_custom_graph_requirements_params = \
            any([k not in default_graph_requirements_params_dict for k in self._input_params])
        for k, v in self._input_params.items():
            # add all parameters except common left unused after GPAlgorithmParameters and GraphGenerationParams
            # initialization, since it can be custom domain specific params
            if k not in self._default_common_params:
                default_graph_requirements_params_dict[k] = self._input_params[k]
        if is_custom_graph_requirements_params:
            return DynamicGraphRequirements(default_graph_requirements_params_dict)
        else:
            return GraphRequirements(**default_graph_requirements_params_dict)

    def get_actual_common_params(self) -> Dict[str, Any]:
        for k, v in self._input_params.items():
            if k in self._default_common_params:
                self._default_common_params[k] = v
        return self._default_common_params
