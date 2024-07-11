import dataclasses
import datetime
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional

from golem.core.paths import default_data_dir


@dataclass
class OptimizationParameters:
    """Defines general algorithm-independent parameters of the composition process
    (like stop condition, validation, timeout, logging etc.)

    Options related to stop condition:

    :param num_of_generations: maximum number of optimizer generations
    :param timeout: max time in minutes available for composition process
    :param early_stopping_iterations: for early stopping.

        Optional max number of stagnating
        iterations for early stopping. If both early_stopping options are None,
        then do not use early stopping.

    :param early_stopping_timeout: for early stopping.

        Optional duration (in minutes) of stagnating
        optimization for early stopping. If both early_stopping options are None,
        then do not use early stopping.

    Infrastructure options (logging, performance)

    :param keep_n_best: number of the best individuals of previous generation to keep in next generation
    :param max_graph_fit_time: time constraint for evaluation of each graph (datetime.timedelta)
    :param n_jobs: num of n_jobs
    :param show_progress: bool indicating whether to show progress using tqdm or not
    :param collect_intermediate_metric: save metrics for intermediate (non-root) nodes in graph
    :param parallelization_mode: identifies the way to parallelize population evaluation

    History options:

    :param keep_history: if True, then save generations to history; if False, don't keep history.
    :param history_dir: directory for saving optimization history, optional.

        If the path is relative, then save relative to `default_data_dir`.
        If absolute -- then save directly by specified path.
        If None -- do not save the history to disk and keep it only in-memory.
    """

    num_of_generations: Optional[int] = None
    timeout: Optional[datetime.timedelta] = datetime.timedelta(minutes=5)
    early_stopping_iterations: Optional[int] = 50
    early_stopping_timeout: Optional[float] = 5

    keep_n_best: int = 1
    max_graph_fit_time: Optional[datetime.timedelta] = None
    n_jobs: int = 1
    show_progress: bool = True
    collect_intermediate_metric: bool = False
    parallelization_mode: str = 'populational'
    static_individual_metadata: dict = field(default_factory=lambda: {
        'use_input_preprocessing': True
    })

    keep_history: bool = True
    history_dir: Optional[str] = field(default_factory=default_data_dir)
    agent_dir: Optional[str] = field(default_factory=default_data_dir)


@dataclass
class GraphRequirements(OptimizationParameters):
    """Defines restrictions and requirements on final graphs.

    Restrictions on final graphs:

    :param start_depth: start value of adaptive tree depth
    :param max_depth: max depth of the resulting graph
    :param min_arity: min number of parents for node
    :param max_arity: max number of parents for node
    """

    start_depth: int = 3
    max_depth: int = 10
    min_arity: int = 2
    max_arity: int = 4

    def __post_init__(self):
        excluded_fields = ['n_jobs']
        for field_name, field_value in dataclasses.asdict(self).items():
            if field_name in excluded_fields:
                continue
            if isinstance(field_value, Number) and field_value < 0:
                raise ValueError(f'Value of {field_name} must be non-negative')
