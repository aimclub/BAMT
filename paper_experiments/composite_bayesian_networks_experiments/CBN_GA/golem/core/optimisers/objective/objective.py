import itertools
from dataclasses import dataclass
from numbers import Real
from typing import Any, Optional, Callable, Sequence, TypeVar, Dict, Tuple, Union

from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness, SingleObjFitness, null_fitness, MultiObjFitness

G = TypeVar('G', bound=Graph, covariant=True)
R = TypeVar('R', contravariant=True)
GraphFunction = Callable[[G], R]
ObjectiveFunction = GraphFunction[G, Fitness]


@dataclass
class ObjectiveInfo:
    """Keeps information about used metrics."""
    is_multi_objective: bool = False
    metric_names: Sequence[str] = ()

    def format_fitness(self, fitness: Union[Fitness, Sequence[float]]) -> str:
        """Returns formatted fitness string.
        Example for 3 metrics: `<roc_auc=0.542 f1=0.72 complexity=0.8>`"""
        values = fitness.values if isinstance(fitness, Fitness) else fitness
        fitness_info_str = [f'{name}={value:.3f}'
                            if value is not None
                            else f'{name}=None'
                            for name, value in zip(self.metric_names, values)]
        return f"<{' '.join(fitness_info_str)}>"


class Objective(ObjectiveInfo, ObjectiveFunction):
    """Represents objective function for computing metric values
    on Graphs and keeps information about metrics used."""

    def __init__(self,
                 quality_metrics: Union[Callable, Dict[Any, Callable]],
                 complexity_metrics: Optional[Dict[Any, Callable]] = None,
                 is_multi_objective: bool = False,
                 ):
        self._log = default_log(self)
        if isinstance(quality_metrics, Callable):
            quality_metrics = {'metric': quality_metrics}
        self.quality_metrics = quality_metrics
        self.complexity_metrics = complexity_metrics or {}
        metric_names = [str(metric_id) for metric_id, _ in self.metrics]
        ObjectiveInfo.__init__(self, is_multi_objective, metric_names)

    def __call__(self, graph: Graph, **metrics_kwargs: Any) -> Fitness:
        evaluated_metrics = []
        for metric_id, metric_func in self.metrics:
            try:
                metric_value = metric_func(graph, **metrics_kwargs)
                evaluated_metrics.append(metric_value)
            except Exception as ex:
                self._log.error(f'Objective evaluation error for graph {graph} on metric {metric_id}: {ex}')
                return null_fitness()  # fail right away
        return to_fitness(evaluated_metrics, self.is_multi_objective)

    @property
    def metrics(self) -> Sequence[Tuple[Any, Callable]]:
        return list(itertools.chain(self.quality_metrics.items(), self.complexity_metrics.items()))

    def get_info(self) -> ObjectiveInfo:
        return ObjectiveInfo(self.is_multi_objective, self.metric_names)


def to_fitness(metric_values: Optional[Sequence[Real]], multi_objective: bool = False) -> Fitness:
    if metric_values is None:
        return null_fitness()
    elif multi_objective:
        return MultiObjFitness(values=metric_values, weights=1.)
    else:
        return SingleObjFitness(*metric_values)


def get_metric_position(metrics, metric_type):
    for num, metric in enumerate(metrics):
        if isinstance(metric, metric_type):
            return num
