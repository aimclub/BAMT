from abc import abstractmethod
from random import choice
from typing import Any, Optional, Sequence, Dict

from golem.core.constants import MIN_POP_SIZE
from golem.core.dag.graph import Graph
from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher, SequentialDispatcher
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.objective import GraphFunction, ObjectiveFunction
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, AlgorithmParameters
from golem.core.optimisers.timer import OptimisationTimer
from golem.utilities.grouped_condition import GroupedCondition


class PopulationalOptimizer(GraphOptimizer):
    """
    Base class of populational optimizer.
    PopulationalOptimizer implements all basic methods for optimization not related to evolution process
    to experiment with other kinds of evolution optimization methods
    It allows to find the optimal solution using specified metric (one or several).
    To implement the specific evolution strategy, implement `_evolution_process`.

    Args:
         objective: objective for optimization
         initial_graphs: graphs which were initialized outside the optimizer
         requirements: implementation-independent requirements for graph optimizer
         graph_generation_params: parameters for new graph generation
         graph_optimizer_params: parameters for specific implementation of graph optimizer

    Additional custom params can be specified with `custom_optimizer_params`.
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: Optional['AlgorithmParameters'] = None,
                 **custom_optimizer_params
                 ):
        super().__init__(objective, initial_graphs, requirements,
                         graph_generation_params, graph_optimizer_params, **custom_optimizer_params)
        self.population = None
        self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)

        dispatcher_type = MultiprocessingDispatcher if self.requirements.parallelization_mode == 'populational' else \
            SequentialDispatcher

        self.eval_dispatcher = dispatcher_type(adapter=graph_generation_params.adapter,
                                               n_jobs=requirements.n_jobs,
                                               graph_cleanup_fn=_try_unfit_graph,
                                               delegate_evaluator=graph_generation_params.remote_evaluator)

        # early_stopping_iterations and early_stopping_timeout may be None, so use some obvious max number
        max_stagnation_length = requirements.early_stopping_iterations or requirements.num_of_generations
        max_stagnation_time = requirements.early_stopping_timeout or self.timer.timeout
        self.stop_optimization = \
            GroupedCondition(results_as_message=True).add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_generation_num - 1),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: (requirements.num_of_generations is not None and
                         self.current_generation_num >= requirements.num_of_generations + 1),
                'Optimisation stopped: Max number of generations reached'
            ).add_condition(
                lambda: (max_stagnation_length is not None and
                         self.generations.stagnation_iter_count >= max_stagnation_length),
                'Optimisation finished: Early stopping iterations criteria was satisfied'
            ).add_condition(
                lambda: self.generations.stagnation_time_duration >= max_stagnation_time,
                'Optimisation finished: Early stopping timeout criteria was satisfied'
            )
        # in how many generations structural diversity check should be performed
        self.gen_structural_diversity_check = self.graph_optimizer_params.structural_diversity_frequency_check

    @property
    def current_generation_num(self) -> int:
        return self.generations.generation_num

    def set_evaluation_callback(self, callback: Optional[GraphFunction]):
        # Redirect callback to evaluation dispatcher
        self.eval_dispatcher.set_graph_evaluation_callback(callback)

    def optimise(self, objective: ObjectiveFunction) -> Sequence[Graph]:

        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective, self.timer)

        with self.timer, self._progressbar as pbar:

            self._initial_population(evaluator)

            while not self.stop_optimization():
                try:
                    new_population = self._evolve_population(evaluator)
                    if self.gen_structural_diversity_check != -1 \
                            and self.generations.generation_num % self.gen_structural_diversity_check == 0 \
                            and self.generations.generation_num != 0:
                        new_population = self.get_structure_unique_population(new_population, evaluator)
                    pbar.update()
                except EvaluationAttemptsError as ex:
                    self.log.warning(f'Composition process was stopped due to: {ex}')
                    break
                # Adding of new population to history
                self._update_population(new_population)
        pbar.close()
        self._update_population(self.best_individuals, 'final_choices')
        return [ind.graph for ind in self.best_individuals]

    @property
    def best_individuals(self):
        return self.generations.best_individuals

    @abstractmethod
    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        raise NotImplementedError()

    @abstractmethod
    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """
        raise NotImplementedError()

    def _extend_population(self, pop: PopulationT, target_pop_size: int) -> PopulationT:
        """ Extends population to specified `target_pop_size`. """
        n = target_pop_size - len(pop)
        extended_population = list(pop)
        extended_population.extend([Individual(graph=choice(pop).graph) for _ in range(n)])
        return extended_population

    def _update_population(self, next_population: PopulationT, label: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        self.generations.append(next_population)
        if self.requirements.keep_history:
            self._log_to_history(next_population, label, metadata)
        self._iteration_callback(next_population, self)
        self.population = next_population

        self.log.info(f'Generation num: {self.current_generation_num} size: {len(next_population)}')
        self.log.info(f'Best individuals: {str(self.generations)}')
        if self.generations.stagnation_iter_count > 0:
            self.log.info(f'no improvements for {self.generations.stagnation_iter_count} iterations')
            self.log.info(f'spent time: {round(self.timer.minutes_from_start, 1)} min')

    def _log_to_history(self, population: PopulationT, label: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        self.history.add_to_history(population, label, metadata)
        self.history.add_to_archive_history(self.generations.best_individuals)
        if self.requirements.history_dir:
            self.history.save_current_results(self.requirements.history_dir)

    def get_structure_unique_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """ Increases structurally uniqueness of population to prevent stagnation in optimization process.
        Returned population may be not entirely unique, if the size of unique population is lower than MIN_POP_SIZE. """
        unique_population_with_ids = {ind.graph.descriptive_id: ind for ind in population}
        unique_population = list(unique_population_with_ids.values())

        # if size of unique population is too small, then extend it to MIN_POP_SIZE by repeating individuals
        if len(unique_population) < MIN_POP_SIZE:
            unique_population = self._extend_population(pop=unique_population, target_pop_size=MIN_POP_SIZE)
        return evaluator(unique_population)


# TODO: remove this hack (e.g. provide smth like FitGraph with fit/unfit interface)
def _try_unfit_graph(graph: Any):
    if hasattr(graph, 'unfit'):
        graph.unfit()


class EvaluationAttemptsError(Exception):
    """ Number of evaluation attempts exceeded """

    def __init__(self, *args):
        self.message = args[0] or None

    def __str__(self):
        return self.message or 'Too many fitness evaluation errors.'
