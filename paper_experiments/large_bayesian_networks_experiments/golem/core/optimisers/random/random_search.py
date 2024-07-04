from random import choice
from typing import Optional, Sequence

from golem.core.dag.graph import Graph
from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.genetic.evaluation import SequentialDispatcher
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphOptimizer, GraphGenerationParams
from golem.core.optimisers.timer import OptimisationTimer
from golem.utilities.grouped_condition import GroupedCondition


class RandomSearchOptimizer(GraphOptimizer):

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Graph] = None,
                 requirements: Optional[GraphRequirements] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_params: Optional[GPAlgorithmParameters] = None):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.generations = GenerationKeeper(self.objective, keep_n_best=requirements.keep_n_best)
        self.current_iteration_num = 0
        self.best_individual = None
        self.stop_optimization = \
            GroupedCondition(results_as_message=True).add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_iteration_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: requirements.num_of_generations is not None and
                self.current_iteration_num >= requirements.num_of_generations,
                'Optimisation stopped: Max number of iterations reached')

    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:

        dispatcher = SequentialDispatcher(self.graph_generation_params.adapter)
        evaluator = dispatcher.dispatch(objective, self.timer)

        self.current_iteration_num = 0

        with self.timer, self._progressbar as pbar:
            self.best_individual = self._eval_initial_individual(evaluator)
            self._update_best_individual(self.best_individual, 'initial_assumptions')
            while not self.stop_optimization():
                new_individual = self._generate_new_individual()
                evaluator([new_individual])
                self.current_iteration_num += 1
                self._update_best_individual(new_individual)
                pbar.update()
        self._update_best_individual(self.best_individual, 'final_choices')
        pbar.close()
        return [self.best_individual.graph]

    def _update_best_individual(self, new_individual: Individual, label: Optional[str] = None):
        if new_individual.fitness >= self.best_individual.fitness:
            self.best_individual = new_individual

        self.generations.append([new_individual])

        self.log.info(f'Spent time: {round(self.timer.minutes_from_start, 1)} min')
        self.log.info(f'Iteration num {self.current_iteration_num}: '
                      f'Best individuals fitness {str(self.generations)}')

        self.history.add_to_history([new_individual], label)
        self.history.add_to_archive_history(self.generations.best_individuals)

    def _eval_initial_individual(self, evaluator: EvaluationOperator) -> Individual:
        init_ind = Individual(choice(self.initial_graphs)) if self.initial_graphs else self._generate_new_individual()
        evaluator([init_ind])
        return init_ind

    def _generate_new_individual(self) -> Individual:
        new_graph = self.graph_generation_params.random_graph_factory(self.requirements)
        return Individual(new_graph)
