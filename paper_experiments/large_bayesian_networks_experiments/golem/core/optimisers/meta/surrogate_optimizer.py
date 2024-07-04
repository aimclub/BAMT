from typing import Sequence

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.meta.surrogate_evaluator import SurrogateDispatcher
from golem.core.optimisers.meta.surrogate_model import RandomValuesSurrogateModel
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError, _try_unfit_graph


class SurrogateEachNgenOptimizer(EvoGraphOptimizer):
    """
    Surrogate optimizer that uses surrogate model for evaluating part of individuals

    Additionally, we need to pass surrogate_model object
    """
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters,
                 surrogate_model=RandomValuesSurrogateModel(),
                 surrogate_each_n_gen=5
                 ):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.surrogate_model = surrogate_model
        self.surrogate_each_n_gen = surrogate_each_n_gen
        self.surrogate_dispatcher = SurrogateDispatcher(adapter=graph_generation_params.adapter,
                                                        n_jobs=requirements.n_jobs,
                                                        graph_cleanup_fn=_try_unfit_graph,
                                                        delegate_evaluator=graph_generation_params.remote_evaluator,
                                                        surrogate_model=surrogate_model)

    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:
        # eval_dispatcher defines how to evaluate objective on the whole population
        evaluator = self.eval_dispatcher.dispatch(objective, self.timer)
        # surrogate_dispatcher defines how to evaluate objective with surrogate model
        surrogate_evaluator = self.surrogate_dispatcher.dispatch(objective, self.timer)

        with self.timer, self._progressbar:
            self._initial_population(evaluator)
            while not self.stop_optimization():
                try:
                    if self.generations.generation_num % self.surrogate_each_n_gen == 0:
                        new_population = self._evolve_population(surrogate_evaluator)
                    else:
                        new_population = self._evolve_population(evaluator)
                except EvaluationAttemptsError as ex:
                    self.log.warning(f'Composition process was stopped due to: {ex}')
                    break
                # Adding of new population to history
                self._update_population(new_population)
        self._update_population(self.best_individuals, 'final_choices')
        return [ind.graph for ind in self.best_individuals]
