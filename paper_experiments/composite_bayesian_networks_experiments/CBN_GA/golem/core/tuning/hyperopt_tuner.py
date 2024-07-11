from abc import ABC
from datetime import timedelta
from typing import Optional, Callable, Dict

import numpy as np
from hyperopt import tpe, hp
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll import Apply

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.log import default_log
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.optimisers.timer import Timer
from golem.core.tuning.search_space import SearchSpace, get_node_operation_parameter_label
from golem.core.tuning.tuner_interface import BaseTuner


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
                 algo: Callable = tpe.suggest):
        early_stopping_rounds = early_stopping_rounds or max(100, int(np.sqrt(iterations) * 10))
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations,
                         early_stopping_rounds,
                         timeout,
                         n_jobs,
                         deviation)

        self.early_stop_fn = no_progress_loss(iteration_stop_count=self.early_stopping_rounds)
        self.max_seconds = int(timeout.seconds) if timeout is not None else None
        self.algo = algo
        self.log = default_log(self)

    def _update_remaining_time(self, tuner_timer: Timer):
        self.max_seconds = self.max_seconds - tuner_timer.minutes_from_start * 60


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


def get_node_parameters_for_hyperopt(search_space: SearchSpace, node_id: int, operation_name: str) \
        -> Dict[str, Apply]:
    """
    Function for forming dictionary with hyperparameters of the node operation for the ``HyperoptTuner``

    Args:
        search_space: SearchSpace with parameters per operation
        node_id: number of node in graph.nodes list
        operation_name: name of operation in the node

    Returns:
        parameters_dict: dictionary-like structure with labeled hyperparameters
        and their range per operation
    """

    # Get available parameters for current operation
    parameters_list = search_space.get_parameters_for_operation(operation_name)

    parameters_dict = {}
    for parameter_name in parameters_list:
        node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

        # For operation get range where search can be done
        space = get_parameter_hyperopt_space(search_space, operation_name, parameter_name, node_op_parameter_name)

        parameters_dict.update({node_op_parameter_name: space})

    return parameters_dict
