from abc import ABC
from datetime import timedelta
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
from hyperopt import hp, tpe, fmin, Trials
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll import Apply, scope
from hyperopt.pyll_utils import validate_label

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.dag.linked_graph_node import LinkedGraphNode
from golem.core.log import default_log
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.tuning.search_space import SearchSpace, get_node_operation_parameter_label
from golem.core.tuning.tuner_interface import BaseTuner


@validate_label
def hp_randint(label, *args, **kwargs):
    return scope.int(scope.hyperopt_param(label, scope.randint(*args, **kwargs)))


hp.randint = hp_randint


class HyperoptTuner(BaseTuner, ABC):
    """Base class for hyperparameters optimization based on hyperopt library

    Args:
      objective_evaluate: objective to optimize
      adapter: the function for processing of external object that should be optimized
      search_space: SearchSpace instance
      iterations: max number of iterations
      early_stopping_rounds: Optional max number of stagnating iterations for early stopping. If ``None``, will be set
          to ``max(100, int(np.sqrt(iterations) * 10))``.
      timeout: max time for tuning
      n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's)
      deviation: required improvement (in percent) of a metric to return tuned graph.
        By default, ``deviation=0.05``, which means that tuned graph will be returned
        if it's metric will be at least 0.05% better than the initial.
      algo: algorithm for hyperparameters optimization with signature similar to :obj:`hyperopt.tse.suggest`
    """

    def __init__(self, objective_evaluate: ObjectiveFunction,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 early_stopping_rounds: Optional[int] = None,
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 deviation: float = 0.05,
                 algo: Callable = tpe.suggest, **kwargs):
        early_stopping_rounds = early_stopping_rounds or max(100, int(np.sqrt(iterations) * 10))
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations,
                         early_stopping_rounds,
                         timeout,
                         n_jobs,
                         deviation, **kwargs)

        self.early_stop_fn = no_progress_loss(iteration_stop_count=self.early_stopping_rounds)
        self.algo = algo
        self.log = default_log(self)

    def _search_near_initial_parameters(self,
                                        objective,
                                        search_space: dict,
                                        initial_parameters: dict,
                                        trials: Trials,
                                        remaining_time: float,
                                        show_progress: bool = True) -> Tuple[Trials, int]:
        """ Method to search using the search space where parameters initially set for the graph are fixed.
        This allows not to lose results obtained while composition process

        Args:
            graph: graph to be tuned
            search_space: dict with parameters to be optimized and their search spaces
            initial_parameters: dict with initial parameters of the graph
            trials: Trials object to store all the search iterations
            show_progress: shows progress of tuning if True

        Returns:
            trials: Trials object storing all the search trials
            init_trials_num: number of iterations made using the search space with fixed initial parameters
        """
        try_initial_parameters = initial_parameters and self.iterations > 1
        if not try_initial_parameters:
            init_trials_num = 0
            return trials, init_trials_num

        is_init_params_full = len(initial_parameters) == len(search_space)
        if self.iterations < 10 or is_init_params_full:
            init_trials_num = 1
        else:
            init_trials_num = min(int(self.iterations * 0.1), 10)

        # fmin updates trials with evaluation points tried out during the call
        fmin(objective,
             search_space,
             trials=trials,
             algo=self.algo,
             max_evals=init_trials_num,
             show_progressbar=show_progress,
             early_stop_fn=self.early_stop_fn,
             timeout=remaining_time)
        return trials, init_trials_num


def get_parameter_hyperopt_space(search_space: SearchSpace,
                                 operation_name: str,
                                 parameter_name: str,
                                 label: str = 'default') -> Optional[Apply]:
    """
    Function return hyperopt object with search_space from search_space dictionary

    Args:
        search_space: SearchSpace with parameters per operation
        operation_name: name of the operation
        parameter_name: name of hyperparameter of particular operation
        label: label to assign in hyperopt pyll

    Returns:
        parameter range
    """

    # Get available parameters for current operation
    operation_parameters = search_space.parameters_per_operation.get(operation_name)

    if operation_parameters is not None:
        parameter_properties = operation_parameters.get(parameter_name)
        hyperopt_distribution = parameter_properties.get('hyperopt-dist')
        sampling_scope = parameter_properties.get('sampling-scope')
        if hyperopt_distribution == hp.loguniform:
            sampling_scope = [np.log(x) for x in sampling_scope]
        return hyperopt_distribution(label, *sampling_scope)
    else:
        return None


def get_node_parameters_for_hyperopt(search_space: SearchSpace, node_id: int, node: LinkedGraphNode) \
        -> Tuple[Dict[str, Apply], Dict[str, Any]]:
    """
    Function for forming dictionary with hyperparameters of the node operation for the ``HyperoptTuner``

    Args:
        search_space: SearchSpace with parameters per operation
        node_id: number of node in graph.nodes list
        node: node from the graph

    Returns:
        parameters_dict: dictionary-like structure with labeled hyperparameters
        and their range per operation
    """

    # Get available parameters for current operation
    operation_name = node.name
    parameters_list = search_space.get_parameters_for_operation(operation_name)

    parameters_dict = {}
    initial_parameters = {}
    for parameter_name in parameters_list:
        node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

        # For operation get range where search can be done
        space = get_parameter_hyperopt_space(search_space, operation_name, parameter_name, node_op_parameter_name)
        parameters_dict.update({node_op_parameter_name: space})

        if parameter_name in node.parameters:
            initial_parameters.update({node_op_parameter_name: node.parameters[parameter_name]})

    return parameters_dict, initial_parameters
