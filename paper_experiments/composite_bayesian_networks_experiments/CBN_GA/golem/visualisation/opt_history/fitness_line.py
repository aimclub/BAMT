import functools
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, Sequence, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from golem.core.log import default_log
from golem.core.optimisers.fitness import null_fitness
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.paths import default_data_dir
from golem.visualisation.opt_history.arg_constraint_wrapper import ArgConstraintWrapper
from golem.visualisation.opt_history.history_visualization import HistoryVisualization
from golem.visualisation.opt_history.utils import show_or_save_figure

if TYPE_CHECKING:
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


def with_alternate_matplotlib_backend(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        default_mpl_backend = mpl.get_backend()
        try:
            mpl.use('TKAgg')
            return func(*args, **kwargs)
        except ImportError as e:
            default_log(prefix='Requirements').warning(e)
        finally:
            mpl.use(default_mpl_backend)

    return wrapper


def setup_fitness_plot(axis: plt.Axes, xlabel: str, title: Optional[str] = None, with_legend: bool = False):
    if axis is None:
        fig, axis = plt.subplots()

    if with_legend:
        axis.legend()
    axis.set_ylabel('Fitness')
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis='y')


def plot_fitness_line_per_time(axis: plt.Axes, generations, label: Optional[str] = None,
                               with_generation_limits: bool = True) \
        -> Dict[int, Individual]:
    best_fitness = null_fitness()
    gen_start_times = []
    best_individuals = {}

    start_time = datetime.fromisoformat(
        min(generations[0], key=lambda ind: ind.metadata['evaluation_time_iso']).metadata[
            'evaluation_time_iso'])
    end_time_seconds = (datetime.fromisoformat(
        max(generations[-1], key=lambda ind: ind.metadata['evaluation_time_iso']).metadata[
            'evaluation_time_iso']) - start_time).seconds

    for gen_num, gen in enumerate(generations):
        gen_start_times.append(1e10)
        gen_sorted = sorted(gen, key=lambda ind: ind.metadata['evaluation_time_iso'])
        for ind in gen_sorted:
            if ind.native_generation != gen_num:
                continue
            evaluation_time = (datetime.fromisoformat(ind.metadata['evaluation_time_iso']) - start_time).seconds
            if evaluation_time < gen_start_times[gen_num]:
                gen_start_times[gen_num] = evaluation_time
            if ind.fitness > best_fitness:
                best_individuals[evaluation_time] = ind
                best_fitness = ind.fitness

    best_eval_times, best_fitnesses = np.transpose(
        [(evaluation_time, abs(individual.fitness.value))
         for evaluation_time, individual in best_individuals.items()])

    best_eval_times = list(best_eval_times)
    best_fitnesses = list(best_fitnesses)

    if best_eval_times[-1] != end_time_seconds:
        best_fitnesses.append(abs(best_fitness.value))
        best_eval_times.append(end_time_seconds)
    gen_start_times.append(end_time_seconds)

    axis.step(best_eval_times, best_fitnesses, where='post', label=label)

    if with_generation_limits:
        axis_gen = axis.twiny()
        axis_gen.set_xlim(axis.get_xlim())
        axis_gen.set_xticks(gen_start_times, list(range(len(gen_start_times) - 1)) + [''])
        axis_gen.locator_params(nbins=10)
        axis_gen.set_xlabel('Generation')

        gen_ticks = axis_gen.get_xticks()
        prev_time = gen_ticks[0]
        axis.axvline(prev_time, color='k', linestyle='--', alpha=0.3)
        for i, next_time in enumerate(gen_ticks[1:]):
            axis.axvline(next_time, color='k', linestyle='--', alpha=0.3)
            if i % 2 == 0:
                axis.axvspan(prev_time, next_time, color='k', alpha=0.05)
            prev_time = next_time

    return best_individuals


def find_best_running_fitness(generations: Sequence[Sequence[Individual]],
                              metric_id: int = 0,
                              ) -> Tuple[List[float], List[int], Dict[int, Individual]]:
    """For each trial history per each generation find the best fitness *seen so far*.
    Returns tuple:
    - list of best seen metric up to that generation,
    - list of indices where current best individual belongs.
    - dict mapping of best index to best individuals
    """
    best_metric = np.inf  # Assuming metric minimization
    best_individuals = {}

    # Core logic
    for gen_num, gen in enumerate(generations):
        for ind in gen:
            if ind.native_generation != gen_num:
                continue
            target_metric = ind.fitness.values[metric_id]
            if target_metric <= best_metric:
                best_individuals[gen_num] = ind
                best_metric = target_metric

    # Additional unwrapping of the data for simpler plotting
    best_generations, best_metrics = np.transpose(
        [(gen_num, abs(individual.fitness.values[metric_id]))
         for gen_num, individual in best_individuals.items()])
    best_generations = list(best_generations)
    best_metrics = list(best_metrics)
    if best_generations[-1] != len(generations) - 1:
        best_metrics.append(abs(best_metric))
        best_generations.append(len(generations) - 1)

    return best_metrics, best_generations, best_individuals


def plot_fitness_line_per_generations(axis: plt.Axes, generations, label: Optional[str] = None) \
        -> Dict[int, Individual]:
    best_fitnesses, best_generations, best_individuals = find_best_running_fitness(generations, metric_id=0)
    axis.step(best_generations, best_fitnesses, where='post', label=label)
    axis.set_xticks(range(len(generations)))
    axis.locator_params(nbins=10)
    return best_individuals


class FitnessLine(HistoryVisualization):
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None, dpi: Optional[int] = None,
                  per_time: Optional[bool] = None):
        """ Visualizes the best fitness values during the evolution in the form of line.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed.
        :param dpi: DPI of the output figure.
        :param per_time: defines whether to show time grid if it is available in history.
        """
        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')
        per_time = per_time if per_time is not None else self.get_predefined_value('per_time') or False

        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
        if per_time:
            xlabel = 'Time, s'
            plot_fitness_line_per_time(ax, self.history.generations)
        else:
            xlabel = 'Generation'
            plot_fitness_line_per_generations(ax, self.history.generations)
        setup_fitness_plot(ax, xlabel)
        show_or_save_figure(fig, save_path, dpi)


class FitnessLineInteractive(HistoryVisualization):

    @with_alternate_matplotlib_backend
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None, dpi: Optional[int] = None,
                  per_time: Optional[bool] = None, graph_show_kwargs: Optional[Dict[str, Any]] = None):
        """ Visualizes the best fitness values during the evolution in the form of line.
        Additionally, shows the structure of the best individuals and the moment of their discovering.
        :param save_path: path to save the visualization. If set, then the image will be saved, and if not,
            it will be displayed.
        :param dpi: DPI of the output figure.
        :param per_time: defines whether to show time grid if it is available in history.
        :param graph_show_kwargs: keyword arguments of `graph.show()` function.
        """

        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')
        per_time = per_time if per_time is not None else self.get_predefined_value('per_time') or False
        graph_show_kwargs = graph_show_kwargs or self.get_predefined_value('graph_show_params') or {}

        graph_show_kwargs = graph_show_kwargs or self.visuals_params.get('graph_show_params') or {}

        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        ax_fitness, ax_graph = axes

        if per_time:
            x_label = 'Time, s'
            x_template = 'time {} s'
            plot_func = plot_fitness_line_per_time
        else:
            x_label = 'Generation'
            x_template = 'generation {}'
            plot_func = plot_fitness_line_per_generations

        best_individuals = plot_func(ax_fitness, self.history.generations)
        setup_fitness_plot(ax_fitness, x_label)

        ax_graph.axis('off')

        class InteractivePlot:
            temp_path = Path(default_data_dir(), 'current_graph.png')

            def __init__(self, best_individuals: Dict[int, Individual]):
                self.best_x: List[int] = list(best_individuals.keys())
                self.best_individuals: List[Individual] = list(best_individuals.values())
                self.index: int = len(self.best_individuals) - 1
                self.time_line = ax_fitness.axvline(self.best_x[self.index], color='r', alpha=0.7)
                self.graph_images: List[np.ndarray] = []
                self.generate_graph_images()
                self.update_graph()

            def generate_graph_images(self):
                for ind in self.best_individuals:
                    graph = ind.graph
                    graph.show(self.temp_path, **graph_show_kwargs)
                    self.graph_images.append(plt.imread(str(self.temp_path)))
                self.temp_path.unlink()

            def update_graph(self):
                ax_graph.imshow(self.graph_images[self.index])
                x = self.best_x[self.index]
                fitness = self.best_individuals[self.index].fitness
                ax_graph.set_title(f'The best graph at {x_template.format(x)}, fitness={fitness}')

            def update_time_line(self):
                self.time_line.set_xdata(self.best_x[self.index])

            def step_index(self, step: int):
                self.index = (self.index + step) % len(self.best_individuals)
                self.update_graph()
                self.update_time_line()
                plt.draw()

            def next(self, event):
                self.step_index(1)

            def prev(self, event):
                self.step_index(-1)

        callback = InteractivePlot(best_individuals)

        if not save_path:  # display buttons only for an interactive plot
            ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
            ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
            b_next = Button(ax_next, 'Next')
            b_next.on_clicked(callback.next)
            b_prev = Button(ax_prev, 'Previous')
            b_prev.on_clicked(callback.prev)

        show_or_save_figure(fig, save_path, dpi)


class MultipleFitnessLines(metaclass=ArgConstraintWrapper):
    """ Class to compare fitness changes during optimization process.
    :param histories_to_compare: dictionary with labels to display as keys and histories as values. """

    def __init__(self,
                 histories_to_compare: Dict[str, Sequence['OptHistory']],
                 visuals_params: Dict[str, Any] = None):
        self.histories_to_compare = histories_to_compare
        self.visuals_params = visuals_params or {}
        self.log = default_log(self)

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
        setup_fitness_plot(ax, xlabel)
        plt.legend()
        show_or_save_figure(fig, save_path, dpi)

    def plot_multiple_fitness_lines(self, ax: plt.axis, metric_id: int = 0, with_confidence: bool = True):
        for histories, label in zip(list(self.histories_to_compare.values()), list(self.histories_to_compare.keys())):
            plot_average_fitness_line_per_generations(ax, histories, label,
                                                      with_confidence=with_confidence,
                                                      metric_id=metric_id)

    def get_predefined_value(self, param: str):
        return self.visuals_params.get(param)


def plot_average_fitness_line_per_generations(
        axis: plt.Axes,
        histories: Sequence['OptHistory'],
        label: Optional[str] = None,
        metric_id: int = 0,
        with_confidence: bool = True,
        z_score: float = 1.96):
    """Plots average fitness line per number of histories
    with confidence interval for given z-score (default z=1.96 is for 95% confidence)."""

    trial_fitnesses: List[List[float]] = []
    for history in histories:
        best_fitnesses, _, _ = find_best_running_fitness(history.generations, metric_id)
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
