import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.log import Log
from golem.core.optimisers.genetic.evaluation import DelegateEvaluator, SequentialDispatcher
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.meta.surrogate_model import SurrogateModel, RandomValuesSurrogateModel
from golem.core.optimisers.objective.objective import to_fitness, GraphFunction
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult


class SurrogateDispatcher(SequentialDispatcher):
    """Evaluates objective function with surrogate model.
        Usage: call `dispatch(objective_function)` to get evaluation function.
        Additionally, we need to pass surrogate_model object
    """

    def __init__(self,
                 adapter: BaseOptimizationAdapter,
                 n_jobs: int = 1,
                 graph_cleanup_fn: Optional[GraphFunction] = None,
                 delegate_evaluator: Optional[DelegateEvaluator] = None,
                 surrogate_model: SurrogateModel = RandomValuesSurrogateModel()):
        super().__init__(adapter, n_jobs, graph_cleanup_fn, delegate_evaluator)
        self._n_jobs = 1
        self.surrogate_model = surrogate_model

    def evaluate_single(self, graph: OptGraph, uid_of_individual: str, with_time_limit: bool = True,
                        cache_key: Optional[str] = None,
                        logs_initializer: Optional[Tuple[int, pathlib.Path]] = None) -> GraphEvalResult:
        graph = self.evaluation_cache.get(cache_key, graph)
        if logs_initializer is not None:
            # in case of multiprocessing run
            Log.setup_in_mp(*logs_initializer)

        start_time = timeit.default_timer()
        fitness = to_fitness(self.surrogate_model(graph, objective=self._objective_eval))
        end_time = timeit.default_timer()

        eval_res = GraphEvalResult(
            uid_of_individual=uid_of_individual, fitness=fitness, graph=graph, metadata={
                'computation_time_in_seconds': end_time - start_time,
                'evaluation_time_iso': datetime.now().isoformat(),
                'surrogate_evaluation': True
            }
        )
        return eval_res
