from functools import partial
from typing import Tuple, Optional

from hyperopt import Trials, fmin, space_eval

from golem.core.constants import MIN_TIME_FOR_TUNING_IN_SEC
from golem.core.optimisers.graph import OptGraph
from golem.core.tuning.hyperopt_tuner import HyperoptTuner, get_node_parameters_for_hyperopt
from golem.core.tuning.tuner_interface import DomainGraphForTune


class SimultaneousTuner(HyperoptTuner):
    """
        Class for hyperparameters optimization for all nodes simultaneously
    """

    def _tune(self, graph: DomainGraphForTune, show_progress: bool = True) -> DomainGraphForTune:
        """ Function for hyperparameters tuning on the entire graph

        Args:
            graph: graph which hyperparameters will be tuned
            show_progress: shows progress of tuning if True

        Returns:
            Graph with tuned hyperparameters
        """
        parameters_dict, init_parameters = self._get_parameters_for_tune(graph)
        remaining_time = self._get_remaining_time()

        if self._check_if_tuning_possible(graph, len(parameters_dict) > 0, remaining_time):
            trials = Trials()

            try:
                # try searching using initial parameters
                # (uses original search space with fixed initial parameters)
                trials, init_trials_num = self._search_near_initial_parameters(
                    partial(self._objective,
                            graph=graph,
                            unchangeable_parameters=init_parameters),
                    parameters_dict,
                    init_parameters,
                    trials,
                    remaining_time,
                    show_progress)
                remaining_time = self._get_remaining_time()
                if remaining_time > MIN_TIME_FOR_TUNING_IN_SEC:
                    fmin(partial(self._objective, graph=graph),
                         parameters_dict,
                         trials=trials,
                         algo=self.algo,
                         max_evals=self.iterations,
                         show_progressbar=show_progress,
                         early_stop_fn=self.early_stop_fn,
                         timeout=remaining_time)
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
        else:
            final_graph = graph
        return final_graph

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
            # Assign unique prefix for each model hyperparameter
            # label - number of node in the graph
            tunable_node_params, initial_node_params = get_node_parameters_for_hyperopt(self.search_space,
                                                                                        node_id=node_id,
                                                                                        node=node)
            parameters_dict.update(tunable_node_params)
            initial_parameters.update(initial_parameters)

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
