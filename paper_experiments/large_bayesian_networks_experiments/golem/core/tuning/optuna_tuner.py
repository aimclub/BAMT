from copy import deepcopy
from datetime import timedelta
from functools import partial
from typing import Optional, Tuple, Union, Sequence

import optuna
from optuna import Trial, Study
from optuna.trial import FrozenTrial

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.tuning.search_space import SearchSpace, get_node_operation_parameter_label
from golem.core.tuning.tuner_interface import BaseTuner, DomainGraphForTune
from golem.utilities.data_structures import ensure_wrapped_in_sequence


class OptunaTuner(BaseTuner):
    def __init__(self, objective_evaluate: ObjectiveFunction,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 early_stopping_rounds: Optional[int] = None,
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 deviation: float = 0.05, **kwargs):
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations,
                         early_stopping_rounds,
                         timeout,
                         n_jobs,
                         deviation, **kwargs)
        self.study = None

    def _tune(self, graph: DomainGraphForTune, show_progress: bool = True) -> \
            Union[DomainGraphForTune, Sequence[DomainGraphForTune]]:
        predefined_objective = partial(self.objective, graph=graph)

        self.objectives_number = len(ensure_wrapped_in_sequence(self.init_metric))
        is_multi_objective = self.objectives_number > 1

        self.study = optuna.create_study(directions=['minimize'] * self.objectives_number)

        init_parameters, has_parameters_to_optimize = self._get_initial_point(graph)
        remaining_time = self._get_remaining_time()
        if self._check_if_tuning_possible(graph,
                                          has_parameters_to_optimize,
                                          remaining_time,
                                          supports_multi_objective=True):
            # Enqueue initial point to try
            if init_parameters:
                self.study.enqueue_trial(init_parameters)

            verbosity_level = optuna.logging.INFO if show_progress else optuna.logging.WARNING
            optuna.logging.set_verbosity(verbosity_level)

            self.study.optimize(predefined_objective,
                                n_trials=self.iterations,
                                n_jobs=self.n_jobs,
                                timeout=remaining_time,
                                callbacks=[self.early_stopping_callback] if not is_multi_objective else None,
                                show_progress_bar=show_progress)

            if not is_multi_objective:
                best_parameters = self.study.best_trials[0].params
                tuned_graphs = self.set_arg_graph(graph, best_parameters)
                self.was_tuned = True
            else:
                tuned_graphs = []
                for best_trial in self.study.best_trials:
                    best_parameters = best_trial.params
                    tuned_graph = self.set_arg_graph(deepcopy(graph), best_parameters)
                    tuned_graphs.append(tuned_graph)
                    self.was_tuned = True
        else:
            tuned_graphs = graph
        return tuned_graphs

    def objective(self, trial: Trial, graph: OptGraph) -> Union[float, Sequence[float, ]]:
        new_parameters = self._get_parameters_from_trial(graph, trial)
        new_graph = BaseTuner.set_arg_graph(graph, new_parameters)
        metric_value = self.get_metric_value(new_graph)
        return metric_value

    def _get_parameters_from_trial(self, graph: OptGraph, trial: Trial) -> dict:
        new_parameters = {}
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Get available parameters for operation
            tunable_node_params = self.search_space.parameters_per_operation.get(operation_name, {})

            for parameter_name, parameter_properties in tunable_node_params.items():
                node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

                parameter_type = parameter_properties.get('type')
                sampling_scope = parameter_properties.get('sampling-scope')
                if parameter_type == 'discrete':
                    new_parameters.update({node_op_parameter_name:
                                           trial.suggest_int(node_op_parameter_name, *sampling_scope)})
                elif parameter_type == 'continuous':
                    new_parameters.update({node_op_parameter_name:
                                           trial.suggest_float(node_op_parameter_name, *sampling_scope)})
                elif parameter_type == 'categorical':
                    new_parameters.update({node_op_parameter_name:
                                           trial.suggest_categorical(node_op_parameter_name, *sampling_scope)})
        return new_parameters

    def _get_initial_point(self, graph: OptGraph) -> Tuple[dict, bool]:
        initial_parameters = {}
        has_parameters_to_optimize = False
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Get available parameters for operation
            tunable_node_params = self.search_space.parameters_per_operation.get(operation_name)

            if tunable_node_params:
                has_parameters_to_optimize = True
                tunable_initial_params = {get_node_operation_parameter_label(node_id, operation_name, p):
                                          node.parameters[p] for p in node.parameters if p in tunable_node_params}
                if tunable_initial_params:
                    initial_parameters.update(tunable_initial_params)
        return initial_parameters, has_parameters_to_optimize

    def early_stopping_callback(self, study: Study, trial: FrozenTrial):
        if self.early_stopping_rounds is not None:
            current_trial_number = trial.number
            best_trial_number = study.best_trial.number
            should_stop = (current_trial_number - best_trial_number) >= self.early_stopping_rounds
            if should_stop:
                self.log.debug('Early stopping rounds criteria was reached')
                study.stop()
