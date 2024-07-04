from copy import deepcopy
from datetime import timedelta
from functools import partial
from typing import Callable, Optional

from hyperopt import tpe, fmin, space_eval, Trials

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.constants import MIN_TIME_FOR_TUNING_IN_SEC
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveFunction
from golem.core.tuning.hyperopt_tuner import HyperoptTuner, get_node_parameters_for_hyperopt
from golem.core.tuning.search_space import SearchSpace
from golem.core.tuning.tuner_interface import DomainGraphForTune


class SequentialTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes sequentially
    """

    def __init__(self, objective_evaluate: ObjectiveFunction,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 early_stopping_rounds: Optional[int] = None,
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 deviation: float = 0.05,
                 algo: Callable = tpe.suggest,
                 inverse_node_order: bool = False, **kwargs):
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations,
                         early_stopping_rounds, timeout,
                         n_jobs,
                         deviation,
                         algo, **kwargs)

        self.inverse_node_order = inverse_node_order

    def _tune(self, graph: DomainGraphForTune, **kwargs) -> DomainGraphForTune:
        """ Method for hyperparameters tuning on the entire graph

        Args:
            graph: graph which hyperparameters will be tuned
        """
        remaining_time = self._get_remaining_time()
        if self._check_if_tuning_possible(graph, parameters_to_optimize=True, remaining_time=remaining_time):
            # Calculate amount of iterations we can apply per node
            nodes_amount = graph.length
            iterations_per_node = round(self.iterations / nodes_amount)
            iterations_per_node = int(iterations_per_node)
            if iterations_per_node == 0:
                iterations_per_node = 1

            # Calculate amount of seconds we can apply per node
            if remaining_time is not None:
                seconds_per_node = round(remaining_time / nodes_amount)
                seconds_per_node = int(seconds_per_node)
            else:
                seconds_per_node = None

            # Tuning performed sequentially for every node - so get ids of nodes
            nodes_ids = self.get_nodes_order(nodes_number=nodes_amount)
            final_graph = deepcopy(self.init_graph)
            best_metric = self.init_metric
            for node_id in nodes_ids:
                node = graph.nodes[node_id]

                # Get node's parameters to optimize
                node_params, init_params = get_node_parameters_for_hyperopt(self.search_space, node_id, node)
                if not node_params:
                    self.log.info(f'"{node.name}" operation has no parameters to optimize')
                else:
                    # Apply tuning for current node
                    graph, metric = self._optimize_node(node_id=node_id,
                                                        graph=graph,
                                                        node_params=node_params,
                                                        init_params=init_params,
                                                        iterations_per_node=iterations_per_node,
                                                        seconds_per_node=seconds_per_node)
                    if metric <= best_metric:
                        final_graph = deepcopy(graph)
                        best_metric = metric
            self.was_tuned = True
        return final_graph

    def get_nodes_order(self, nodes_number: int) -> range:
        """ Method returns list with indices of nodes in the graph

        Args:
            nodes_number: number of nodes to get
        """

        if self.inverse_node_order is True:
            # From source data to output
            nodes_ids = range(nodes_number - 1, -1, -1)
        else:
            # From output to source data
            nodes_ids = range(0, nodes_number)

        return nodes_ids

    def tune_node(self, graph: DomainGraphForTune, node_index: int) -> DomainGraphForTune:
        """ Method for hyperparameters tuning for particular node

        Args:
            graph: graph which contains a node to be tuned
            node_index: Index of the node to tune

        Returns:
            Graph with tuned parameters in node with specified index
        """
        graph = self.adapter.adapt(graph)

        with self.timer:
            self.init_check(graph)

            node = graph.nodes[node_index]

            # Get node's parameters to optimize
            node_params, init_params = get_node_parameters_for_hyperopt(self.search_space,
                                                                        node_id=node_index,
                                                                        node=node)

            remaining_time = self._get_remaining_time()
            if self._check_if_tuning_possible(graph, len(node_params) > 1, remaining_time):
                # Apply tuning for current node
                graph, _ = self._optimize_node(graph=graph,
                                               node_id=node_index,
                                               node_params=node_params,
                                               init_params=init_params,
                                               iterations_per_node=self.iterations,
                                               seconds_per_node=remaining_time
                                               )

                self.was_tuned = True

                # Validation is the optimization do well
                final_graph = self.final_check(graph)
            else:
                final_graph = graph
                self.obtained_metric = self.init_metric
        final_graph = self.adapter.restore(final_graph)
        return final_graph

    def _optimize_node(self, graph: OptGraph,
                       node_id: int,
                       node_params: dict,
                       init_params: dict,
                       iterations_per_node: int,
                       seconds_per_node: float) -> OptGraph:
        """
        Method for node optimization

        Args:
            graph: Graph which node is optimized
            node_id: id of the current node in the graph
            node_params: dictionary with parameters for node
            iterations_per_node: amount of iterations to produce
            seconds_per_node: amount of seconds to produce

        Returns:
            updated graph with tuned parameters in particular node
        """
        remaining_time = self._get_remaining_time()
        trials = Trials()
        trials, init_trials_num = self._search_near_initial_parameters(partial(self._objective,
                                                                               graph=graph,
                                                                               node_id=node_id,
                                                                               unchangeable_parameters=init_params),
                                                                       node_params,
                                                                       init_params,
                                                                       trials,
                                                                       remaining_time)

        remaining_time = self._get_remaining_time()
        if remaining_time > MIN_TIME_FOR_TUNING_IN_SEC:
            fmin(partial(self._objective, graph=graph, node_id=node_id),
                 node_params,
                 trials=trials,
                 algo=self.algo,
                 max_evals=iterations_per_node,
                 early_stop_fn=self.early_stop_fn,
                 timeout=seconds_per_node)

        best_params = space_eval(space=node_params, hp_assignment=trials.argmin)
        is_best_trial_with_init_params = trials.best_trial.get('tid') in range(init_trials_num)
        if is_best_trial_with_init_params:
            best_params = {**best_params, **init_params}
        # Set best params for this node in the graph
        graph = self.set_arg_node(graph=graph, node_id=node_id, node_params=best_params)
        return graph, trials.best_trial['result']['loss']

    def _objective(self,
                   node_params: dict,
                   graph: OptGraph,
                   node_id: int,
                   unchangeable_parameters: Optional[dict] = None) -> float:
        """ Objective function for minimization problem

        Args:
            node_params: dictionary with parameters for node
            graph: graph to evaluate
            node_id: id of the node to which parameters should be assigned

        Returns:
            value of objective function
        """
        # replace new parameters with parameters
        if unchangeable_parameters:
            node_params = {**node_params, **unchangeable_parameters}

        # Set hyperparameters for node
        graph = self.set_arg_node(graph=graph, node_id=node_id, node_params=node_params)

        metric_value = self.get_metric_value(graph=graph)
        return metric_value
