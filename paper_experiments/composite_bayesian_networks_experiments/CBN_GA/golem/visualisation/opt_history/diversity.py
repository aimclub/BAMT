import os
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.visualisation.opt_history.history_visualization import HistoryVisualization
from golem.visualisation.opt_history.utils import show_or_save_figure

if TYPE_CHECKING:
    from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


class DiversityLine(HistoryVisualization):
    def visualize(self, show: bool = True, save_path: Optional[Union[os.PathLike, str]] = None):
        """Plots double graphic to estimate diversity of populations during optimization.
        It plots standard deviation of all metrics (y-axis) by generation number (x-axis).
        Additional line show ratio of structurally unique individuals."""
        return plot_diversity_dynamic(self.history, show=show, save_path=save_path, dpi=self.visuals_params['dpi'])


class DiversityPopulation(HistoryVisualization):
    def visualize(self, save_path: Union[os.PathLike, str], fps: int = 4):
        """Creates a GIF with violin-plot estimating distribution of each metric in populations.
        Each frame shows distribution for a particular generation."""
        return plot_diversity_dynamic_gif(self.history, filename=save_path, fps=fps, dpi=self.visuals_params['dpi'])


def compute_fitness_diversity(population: PopulationT) -> np.ndarray:
    """Returns numpy array of standard deviations of fitness values."""
    # substitutes None values
    fitness_values = np.array([ind.fitness.values for ind in population], dtype=float)
    # compute std along each axis while ignoring nan-s
    diversity = np.nanstd(fitness_values, axis=0)
    return diversity


def plot_diversity_dynamic_gif(history: 'OptHistory',
                               filename: Optional[str] = None,
                               fig_size: int = 5,
                               fps: int = 4,
                               dpi: int = 100,
                               ) -> FuncAnimation:
    metric_names = history.objective.metric_names
    # dtype=float removes None, puts np.nan
    # indexed by [population, metric, individual] after transpose (.T)
    pops = history.generations[1:-1]  # ignore initial pop and final choices
    fitness_distrib = [np.array([ind.fitness.values for ind in pop], dtype=float).T
                       for pop in pops]

    # Define bounds on metrics: find min & max on a flattened view of array
    q = 0.025
    lims_max = np.max([np.quantile(pop, 1 - q, axis=1) for pop in fitness_distrib], axis=0)
    lims_min = np.min([np.quantile(pop, q, axis=1) for pop in fitness_distrib], axis=0)

    # Setup the plot
    ncols = max(len(metric_names), 1)
    fig, axs = plt.subplots(ncols=ncols)
    fig.set_size_inches(fig_size * ncols, fig_size)
    axs = np.atleast_1d(np.ravel(axs))

    # Set update function for updating data on the axes
    def update_axes(iframe: int):
        for i, (ax, metric_distrib) in enumerate(zip(axs, fitness_distrib[iframe])):
            # Clear & Prepare axes
            ax: plt.Axes
            ax.clear()
            ax.set_xlim(0.5, 1.5)
            ax.set_ylim(lims_min[i], lims_max[i])
            ax.set_ylabel('Metric value')
            ax.grid()
            # Plot information
            fig.suptitle(f'Population {iframe+1} diversity by metric')
            metric_name = metric_names[i] if metric_names else f"metric{i}"
            ax.set_title(f'{metric_name}, '
                         f'mean={np.mean(metric_distrib).round(3)}, '
                         f'std={np.nanstd(metric_distrib).round(3)}')
            ax.violinplot(metric_distrib,
                          quantiles=[0.25, 0.5, 0.75])

    # Run this function in FuncAnimation
    num_frames = len(fitness_distrib)
    animate = FuncAnimation(
        fig=fig,
        func=update_axes,
        save_count=num_frames,
        interval=200,
    )
    # Save the GIF from animation
    if filename:
        animate.save(filename, fps=fps, dpi=dpi)
    return animate


def plot_diversity_dynamic(history: 'OptHistory',
                           show: bool = True, save_path: Optional[str] = None, dpi: int = 100):
    labels = history.objective.metric_names
    h = history.generations[:-1]  # don't consider final choices
    xs = np.arange(len(h))

    # Compute diversity by metrics
    np_history = np.array([compute_fitness_diversity(pop) for pop in h])
    ys = {label: np_history[:, i] for i, label in enumerate(labels)}
    # Compute number of unique individuals, plot
    ratio_unique = [len(set(ind.graph.descriptive_id for ind in pop)) / len(pop) for pop in h]

    fig, ax = plt.subplots()
    fig.suptitle('Population diversity')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Metrics Std')
    ax.grid()
    line_alpha = 0.8

    for label, metric_std in ys.items():
        ax.plot(xs, metric_std, label=label, alpha=line_alpha)

    ax2 = ax.twinx()
    ax2_color = 'm'  # magenta
    ax2.set_ylabel('Structural uniqueness', color=ax2_color)
    ax2.set_ylim(0.25, 1.05)
    ax2.tick_params(axis='y', labelcolor=ax2_color)
    ax2.plot(xs, ratio_unique, label='unique ratio',
             color=ax2_color, linestyle='dashed', alpha=line_alpha)

    # ask matplotlib for the plotted objects and their labels
    # to put them into single legend for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,
               loc='upper left', bbox_to_anchor=(0., 1.15))

    if show or save_path:
        show_or_save_figure(fig, save_path, dpi)
