from functools import partial
from typing import Tuple, Optional

from hyperopt import Trials, fmin, space_eval

from golem.core.constants import MIN_TIME_FOR_TUNING_IN_SEC
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.timer import Timer
from golem.core.tuning.hyperopt_tuner import HyperoptTuner, get_node_parameters_for_hyperopt
from golem.core.tuning.search_space import get_node_operation_parameter_label
from golem.core.tuning.tuner_interface import DomainGraphForTune


class SimultaneousTuner(HyperoptTuner):
    """
        Class for hyperparameters optimization for all nodes simultaneously
    """

    def tune(self, graph: DomainGraphForTune, show_progress: bool = True) -> DomainGraphForTune:
        """ Function for hyperparameters tuning on the entire graph

        Args:
            graph: graph which hyperparameters will be tuned
            show_progress: shows progress of tuning if True

        Returns:
            Graph with tuned hyperparameters
        """

        graph = self.adapter.adapt(graph)
        parameters_dict, init_parameters = self._get_parameters_for_tune(graph)

        with Timer() as global_tuner_timer:
            self.init_check(graph)
            self._update_remaining_time(global_tuner_timer)

            if not parameters_dict:
                self._stop_tuning_with_message(f'Graph "{graph.graph_description}" has no parameters to optimize')
                final_graph = graph

            elif self.max_seconds <= MIN_TIME_FOR_TUNING_IN_SEC:
                self._stop_tuning_with_message('Tunner stopped after initial assumption due to the lack of time')
                final_graph = graph

            else:
                trials = Trials()

                try:
                    # try searching using initial parameters
                    # (uses original search space with fixed initial parameters)
                    trials, init_trials_num = self._search_near_initial_parameters(graph,
                                                                                   parameters_dict,
                                                                                   init_parameters,
                                                                                   trials,
                                                                                   show_progress)
                    self._update_remaining_time(global_tuner_timer)
                    if self.max_seconds > MIN_TIME_FOR_TUNING_IN_SEC:
                        fmin(partial(self._objective, graph=graph),
                             parameters_dict,
                             trials=trials,
                             algo=self.algo,
                             max_evals=self.iterations,
                             show_progressbar=show_progress,
                             early_stop_fn=self.early_stop_fn,
                             timeout=self.max_seconds)
                    else:
                        self.log.message('Tunner stopped after initial search due to the lack of time')

                    best = space_eval(space=parameters_dict, hp_assignment=trials.argmin)
                    # check if best point was obtained using search space with fixed initial parameters
                    is_best_trial_with_init_params = trials.best_trial.get('tid') in range(init_trials_num)
                    if is_best_trial_with_init_params:
                        best = {**best, **init_parameters}

                    final_graph = self.set_arg_graph(graph=graph, parameters=best)

                    self.was_tuned = True

                except Exception as ex:
                    self.log.warning(f'Exception {ex} occurred during tuning')
                    final_graph = graph

        # Validate if optimisation did well
        graph = self.final_check(final_graph)
        final_graph = self.adapter.restore(graph)
        return final_graph

    def _search_near_initial_parameters(self, graph: OptGraph,
                                        search_space: dict,
                                        initial_parameters: dict,
                                        trials: Trials,
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
        fmin(partial(self._objective, graph=graph, unchangeable_parameters=initial_parameters),
             search_space,
             trials=trials,
             algo=self.algo,
             max_evals=init_trials_num,
             show_progressbar=show_progress,
             early_stop_fn=self.early_stop_fn,
             timeout=self.max_seconds)
        return trials, init_trials_num

    def _get_parameters_for_tune(self, graph: OptGraph) -> Tuple[dict, dict]:
        """ Method for defining the search space

        Args:
            graph: graph to be tuned

        Returns:
            parameters_dict: dict with operation names and parameters
            initial_parameters: dict with initial parameters of the graph
        """

        parameters_dict = {}
        initial_parameters = {}
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the graph
            node_params = get_node_parameters_for_hyperopt(self.search_space, node_id=node_id,
                                                           operation_name=operation_name)
            parameters_dict.update(node_params)

            tunable_node_params = self.search_space.get_parameters_for_operation(operation_name)
            if tunable_node_params:
                tunable_initial_params = {get_node_operation_parameter_label(node_id, operation_name, p):
                                          node.parameters[p] for p in node.parameters if p in tunable_node_params}
                if tunable_initial_params:
                    initial_parameters.update(tunable_initial_params)

        return parameters_dict, initial_parameters

    def _objective(self, parameters_dict: dict, graph: OptGraph, unchangeable_parameters: Optional[dict] = None) \
            -> float:
        """
        Objective function for minimization problem

        Args:
            parameters_dict: dict which contains new graph hyperparameters
            graph: graph to optimize
            unchangeable_parameters: dict with parameters that should not be changed

        Returns:
             metric_value: value of objective function
        """

        # replace new parameters with parameters
        if unchangeable_parameters:
            parameters_dict = {**parameters_dict, **unchangeable_parameters}

        # Set hyperparameters for every node
        graph = self.set_arg_graph(graph, parameters_dict)

        metric_value = self.get_metric_value(graph=graph)

        return metric_value
