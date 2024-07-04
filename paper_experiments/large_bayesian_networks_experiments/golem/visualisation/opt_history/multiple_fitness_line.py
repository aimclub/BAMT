import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np
from matplotlib import pyplot as plt

from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.utilities.data_structures import ensure_wrapped_in_sequence
from golem.visualisation.opt_history.arg_constraint_wrapper import ArgConstraintWrapper
from golem.visualisation.opt_history.fitness_line import setup_fitness_plot
from golem.visualisation.opt_history.utils import show_or_save_figure


class MultipleFitnessLines(metaclass=ArgConstraintWrapper):
    """ Class to compare fitness changes during optimization process.
    :param historical_fitnesses: dictionary with labels to display as keys and list of fintess values as dict values."""

    def __init__(self,
                 historical_fitnesses: Dict[str, Sequence[Sequence[Union[float, Sequence[float]]]]],
                 metric_names: List[str],
                 visuals_params: Dict[str, Any] = None):
        self.historical_fitnesses = historical_fitnesses
        self.metric_names = metric_names
        self.visuals_params = visuals_params or {}
        self.log = default_log(self)

    @staticmethod
    def from_saved_histories(experiment_folders: List[str], root_path: os.PathLike) -> 'MultipleFitnessLines':
        """ Loads histories from specified folders extracting only fitness values
         to not store whole histories in memory.
         Args:
            experiment_folders: names of folders with histories for experiment launches.
            root_path: path to the folder with experiments results
        """
        root = Path(root_path)
        historical_fitnesses = {}
        for exp_name in experiment_folders:
            trials = []
            for history_filename in os.listdir(root / exp_name):
                if history_filename.startswith('history'):
                    history = OptHistory.load(root / exp_name / history_filename)
                    trials.append(history.historical_fitness)
                    print(f"Loaded {history_filename}")
            historical_fitnesses[exp_name] = trials
            print(f'Loaded {len(trials)} trial histories for experiment: {exp_name}')
        metric_names = history.objective.metric_names

        return MultipleFitnessLines(historical_fitnesses, metric_names)

    @staticmethod
    def from_histories(histories_to_compare: Dict[str, Sequence['OptHistory']]) -> 'MultipleFitnessLines':
        """
        Args:
            histories_to_compare: dictionary with labels to display as keys and histories as values."""
        historical_fitnesses = {}
        for key, histories in histories_to_compare.items():
            historical_fitnesses.update({key: [history.historical_fitness for history in histories]})
        metric_names = list(histories_to_compare.values())[0][0].objective.metric_names

        return MultipleFitnessLines(historical_fitnesses, metric_names)

    def visualize(self,
                  save_path: Optional[Union[os.PathLike, str]] = None,
                  with_confidence: bool = True,
                  metric_id: int = 0,
                  dpi: Optional[int] = None):
        """ Visualizes the best fitness values during the evolution in the form of line.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed.
        :param with_confidence: bool param specifying to use confidence interval or not.
        :param metric_id: numeric index of the metric to visualize (for multi-objective opt-n).
        :param dpi: DPI of the output figure.
        """
        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')

        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
        xlabel = 'Generation'
        self.plot_multiple_fitness_lines(ax, metric_id, with_confidence)
        setup_fitness_plot(ax, xlabel, title=f'Fitness lines for {self.metric_names[metric_id]}')
        plt.legend()
        show_or_save_figure(fig, save_path, dpi)

    def plot_multiple_fitness_lines(self, ax: plt.axis, metric_id: int = 0, with_confidence: bool = True):
        for histories, label in zip(list(self.historical_fitnesses.values()), list(self.historical_fitnesses.keys())):
            plot_average_fitness_line_per_generations(ax, histories, label,
                                                      with_confidence=with_confidence,
                                                      metric_id=metric_id)

    def get_predefined_value(self, param: str):
        return self.visuals_params.get(param)


def plot_average_fitness_line_per_generations(
        axis: plt.Axes,
        historical_fitnesses: Sequence[Sequence[Union[float, Sequence[float]]]],
        label: Optional[str] = None,
        metric_id: int = 0,
        with_confidence: bool = True,
        z_score: float = 1.96):
    """Plots average fitness line per number of histories
    with confidence interval for given z-score (default z=1.96 is for 95% confidence)."""

    trial_fitnesses: List[List[float]] = []
    for fitnesses in historical_fitnesses:
        best_fitnesses = get_best_fitness_per_generation(fitnesses, metric_id)
        trial_fitnesses.append(best_fitnesses)

    # Get average fitness value with confidence values
    average_fitness_per_gen = []
    confidence_fitness_per_gen = []
    max_generations = max(len(i) for i in trial_fitnesses)
    for i in range(max_generations):
        all_fitness_gen = []
        for fitnesses in trial_fitnesses:
            if i < len(fitnesses):
                all_fitness_gen.append(fitnesses[i])
            else:
                # if history is too short - repeat the best obtained fitness
                all_fitness_gen.append(fitnesses[-1])
        average_fitness_per_gen.append(mean(all_fitness_gen))
        confidence = stdev(all_fitness_gen) / np.sqrt(len(all_fitness_gen)) \
            if len(all_fitness_gen) >= 2 else 0.
        confidence_fitness_per_gen.append(confidence)

    # Compute confidence interval
    xs = np.arange(len(average_fitness_per_gen))
    ys = np.array(average_fitness_per_gen)
    ci = z_score * np.array(confidence_fitness_per_gen)

    axis.plot(xs, average_fitness_per_gen, label=label)
    if with_confidence:
        axis.fill_between(xs, (ys - ci), (ys + ci), alpha=.2)


def get_best_fitness_per_generation(fitnesses: Sequence[Sequence[Union[float, Sequence[float]]]],
                                    metric_id: int = 0,
                                    ) -> List[float]:
    """Per each generation find the best fitness *seen so far*.
    Returns tuple:
    - list of best seen metric up to that generation
    """
    best_metric = np.inf  # Assuming metric minimization
    best_metrics = []

    for gen_num, gen_fitnesses in enumerate(fitnesses[metric_id]):
        target_metric = min(ensure_wrapped_in_sequence(gen_fitnesses))
        if target_metric <= best_metric:
            best_metric = target_metric
        best_metrics.append(best_metric)

    return best_metrics
