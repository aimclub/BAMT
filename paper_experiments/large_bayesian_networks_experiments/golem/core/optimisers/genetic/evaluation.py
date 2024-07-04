import gc
import logging
import pathlib
import timeit
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from typing import List, Optional, Sequence, Tuple, TypeVar, Dict

from joblib import Parallel, delayed

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.dag.graph import Graph
from golem.core.log import default_log, Log
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import GraphFunction, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.core.optimisers.timer import Timer, get_forever_timer
from golem.utilities.serializable import Serializable
from golem.utilities.memory import MemoryAnalytics
from golem.utilities.utilities import determine_n_jobs

# the percentage of successful evaluations,
# at which evolution is not threatened with stagnation at the moment
STAGNATION_EVALUATION_PERCENTAGE = 0.5

EvalResultsList = List[GraphEvalResult]
G = TypeVar('G', bound=Serializable)


class DelegateEvaluator:
    """Interface for delegate evaluator of graphs."""

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        return False

    @abstractmethod
    def compute_graphs(self, graphs: Sequence[G]) -> Sequence[G]:
        raise NotImplementedError()


class ObjectiveEvaluationDispatcher(ABC):
    """Builder for evaluation operator.
    Takes objective function and decides how to evaluate it over population:
    - defines implementation-specific evaluation policy (e.g. sequential, parallel, async);
    - saves additional metadata (e.g. computation time, intermediate metrics values).
    """

    @abstractmethod
    def dispatch(self, objective: ObjectiveFunction, timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return mapped objective function for evaluating population.

        Args:
            objective: objective function that accepts single individual
            timer: optional timer for stopping the evaluation process

        Returns:
            EvaluationOperator: objective function that accepts whole population
        """
        raise NotImplementedError()

    def set_graph_evaluation_callback(self, callback: Optional[GraphFunction]):
        """Set or reset (with None) post-evaluation callback
        that's called on each graph after its evaluation.

        Args:
            callback: callback to be called on each evaluated graph
        """
        pass

    @staticmethod
    def split_individuals_to_evaluate(individuals: PopulationT) -> Tuple[PopulationT, PopulationT]:
        """Split individuals sequence to evaluated and skipped ones."""
        individuals_to_evaluate = []
        individuals_to_skip = []
        for ind in individuals:
            if ind.fitness.valid:
                individuals_to_skip.append(ind)
            else:
                individuals_to_evaluate.append(ind)
        return individuals_to_evaluate, individuals_to_skip

    @staticmethod
    def apply_evaluation_results(individuals: PopulationT,
                                 evaluation_results: EvalResultsList) -> PopulationT:
        """Applies results of evaluation to the evaluated population.
        Excludes individuals that weren't evaluated."""
        evaluation_results = {res.uid_of_individual: res for res in evaluation_results if res}
        individuals_evaluated = []
        for ind in individuals:
            eval_res = evaluation_results.get(ind.uid)
            if not eval_res:
                continue
            ind.set_evaluation_result(eval_res)
            individuals_evaluated.append(ind)
        return individuals_evaluated


class BaseGraphEvaluationDispatcher(ObjectiveEvaluationDispatcher):
    """Base class for dispatchers that evaluate objective function on population.

    Usage: call `dispatch(objective_function)` to get evaluation function.

    Args:
        adapter: adapter for graphs
        n_jobs: number of jobs for multiprocessing or 1 for no multiprocessing.
        graph_cleanup_fn: function to call after graph evaluation, primarily for memory cleanup.
        delegate_evaluator: delegate graph fitter (e.g. for remote graph fitting before evaluation)
    """

    def __init__(self,
                 adapter: BaseOptimizationAdapter,
                 n_jobs: int = 1,
                 graph_cleanup_fn: Optional[GraphFunction] = None,
                 delegate_evaluator: Optional[DelegateEvaluator] = None):
        self._adapter = adapter
        self._objective_eval = None
        self._cleanup = graph_cleanup_fn
        self._post_eval_callback = None
        self._delegate_evaluator = delegate_evaluator

        self.timer = None
        self.logger = default_log(self)
        self._n_jobs = n_jobs
        self.evaluation_cache = None
        self._reset_eval_cache()

    def dispatch(self, objective: ObjectiveFunction, timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        self._objective_eval = objective
        self.timer = timer or get_forever_timer()
        return self.evaluate_population

    def set_graph_evaluation_callback(self, callback: Optional[GraphFunction]):
        self._post_eval_callback = callback

    def population_evaluation_info(self, pop_size: int, evaluated_pop_size: int):
        """ Shows the amount of successfully evaluated individuals and total number of individuals in population.
         If there are more that 50% of successful evaluations than it's more likely
         there is no problem in optimization process. """
        if pop_size == 0 or evaluated_pop_size / pop_size <= STAGNATION_EVALUATION_PERCENTAGE:
            success_rate = evaluated_pop_size / pop_size if pop_size != 0 else 0
            self.logger.warning(f"{evaluated_pop_size} individuals out of {pop_size} in previous population "
                                f"were evaluated successfully. {success_rate}% "
                                f"is a fairly small percentage of successful evaluation.")
        else:
            self.logger.message(f"{evaluated_pop_size} individuals out of {pop_size} in previous population "
                                f"were evaluated successfully.")

    @abstractmethod
    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        raise NotImplementedError()

    def evaluate_single(self, graph: OptGraph, uid_of_individual: str, with_time_limit: bool = True,
                        cache_key: Optional[str] = None,
                        logs_initializer: Optional[Tuple[int, pathlib.Path]] = None) -> GraphEvalResult:

        graph = self.evaluation_cache.get(cache_key, graph)

        if with_time_limit and self.timer.is_time_limit_reached():
            return None
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        adapted_evaluate = self._adapter.adapt_func(self._evaluate_graph)
        start_time = timeit.default_timer()
        fitness, graph = adapted_evaluate(graph)
        end_time = timeit.default_timer()
        eval_time_iso = datetime.now().isoformat()

        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual, fitness=fitness, graph=graph, metadata={
                'computation_time_in_seconds': end_time - start_time,
                'evaluation_time_iso': eval_time_iso
            }
        )
        return eval_res

    def _evaluate_graph(self, domain_graph: Graph) -> Tuple[Fitness, Graph]:
        fitness = self._objective_eval(domain_graph)

        if self._post_eval_callback:
            self._post_eval_callback(domain_graph)
        if self._cleanup:
            self._cleanup(domain_graph)
        gc.collect()

        return fitness, domain_graph

    def evaluate_with_cache(self, population: PopulationT) -> PopulationT:
        reversed_population = list(reversed(population))
        self._remote_compute_cache(reversed_population)
        evaluated_population = self.evaluate_population(reversed_population)
        self._reset_eval_cache()
        return evaluated_population

    def _reset_eval_cache(self):
        self.evaluation_cache: Dict[str, Graph] = {}

    def _remote_compute_cache(self, population: PopulationT):
        self._reset_eval_cache()
        if self._delegate_evaluator and self._delegate_evaluator.is_enabled:
            self.logger.info('Remote fit used')
            restored_graphs = self._adapter.restore(population)
            computed_graphs = self._delegate_evaluator.compute_graphs(restored_graphs)
            self.evaluation_cache = {ind.uid: graph for ind, graph in zip(population, computed_graphs)}


class MultiprocessingDispatcher(BaseGraphEvaluationDispatcher):
    """Evaluates objective function on population using multiprocessing pool
    and optionally model evaluation cache with RemoteEvaluator.

    Usage: call `dispatch(objective_function)` to get evaluation function.

    Args:
        adapter: adapter for graphs
        n_jobs: number of jobs for multiprocessing or 1 for no multiprocessing.
        graph_cleanup_fn: function to call after graph evaluation, primarily for memory cleanup.
        delegate_evaluator: delegate graph fitter (e.g. for remote graph fitting before evaluation)
    """

    def __init__(self,
                 adapter: BaseOptimizationAdapter,
                 n_jobs: int = 1,
                 graph_cleanup_fn: Optional[GraphFunction] = None,
                 delegate_evaluator: Optional[DelegateEvaluator] = None):

        super().__init__(adapter, n_jobs, graph_cleanup_fn, delegate_evaluator)

    def dispatch(self, objective: ObjectiveFunction, timer: Optional[Timer] = None) -> EvaluationOperator:
        """Return handler to this object that hides all details
        and allows only to evaluate population with provided objective."""
        super().dispatch(objective, timer)
        return self.evaluate_with_cache

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(individuals)
        # Evaluate individuals without valid fitness in parallel.
        n_jobs = determine_n_jobs(self._n_jobs, self.logger)

        parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch="2*n_jobs")
        eval_func = partial(self.evaluate_single, logs_initializer=Log().get_parameters())
        evaluation_results = parallel(delayed(eval_func)(ind.graph, ind.uid) for ind in individuals_to_evaluate)
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, evaluation_results)
        # If there were no successful evals then try once again getting at least one,
        # even if time limit was reached
        successful_evals = individuals_evaluated + individuals_to_skip
        self.population_evaluation_info(evaluated_pop_size=len(successful_evals),
                                        pop_size=len(individuals))
        if not successful_evals:
            for single_ind in individuals:
                evaluation_result = eval_func(single_ind.graph, single_ind.uid, with_time_limit=False)
                successful_evals = self.apply_evaluation_results([single_ind], [evaluation_result])
                if successful_evals:
                    break
        MemoryAnalytics.log(self.logger,
                            additional_info='parallel evaluation of population',
                            logging_level=logging.INFO)
        return successful_evals


class SequentialDispatcher(BaseGraphEvaluationDispatcher):
    """Evaluates objective function on population in sequential way.

        Usage: call `dispatch(objective_function)` to get evaluation function.
    """

    def evaluate_population(self, individuals: PopulationT) -> PopulationT:
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(individuals)
        evaluation_results = [self.evaluate_single(ind.graph, ind.uid) for ind in individuals_to_evaluate]
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, evaluation_results)
        evaluated_population = individuals_evaluated + individuals_to_skip
        return evaluated_population
