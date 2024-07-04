import math
import os
from functools import partial
from typing import Callable, List, Union, Optional

from matplotlib import pyplot as plt, animation

from golem.core.dag.graph import Graph
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.visualisation.graph_viz import GraphVisualizer
from golem.visualisation.opt_history.history_visualization import HistoryVisualization


class GenealogicalPath(HistoryVisualization):
    def visualize(self, graph_dist: Callable[[Graph, Graph], float] = None, target_graph: Graph = None,
                  evolution_time_s: float = 8., hold_result_time_s: float = 2.,
                  save_path: Optional[Union[os.PathLike, str]] = None, show: bool = False):
        """
        Takes the best individual from the resultant generation and traces its genealogical path
        taking the most similar parent each time (or the first parent if no similarity measure is provided).
        That makes the picture more stable (and hence comprehensible) and the evolution process more apparent.

        Saves the result as a GIF with the following layout:
        - target graph (if provided) is displayed on the left,
        - evolving graphs go as the next subplot, they evolve from the first generation to the last,
        - and the fitness plot on the right shows fitness dynamics as the graphs evolve.

        :param graph_dist: a function to measure the distance between two graphs. If not provided, all graphs are
            treated as equally distant.
            Works on optimization graphs, not domain graphs. If your distance metric works on domain graphs,
            adapt it with `adapter.adapt_func(your_metric)`.
        :param target_graph: the graph to compare the genealogical path with. Again, optimization graph is expected.
            If provided, it will be displayed on the left throughout the animation.
        :param save_path: path to save the visualization (won't be saved if it's None).
            GIF of video extension is expected.
        :param show: whether to show the visualization.
        :param evolution_time_s: time in seconds for the part of the animation where the evolution process is shown.
        :param hold_result_time_s: time in seconds for the part of the animation where the final result is shown.
        """
        # Treating all graphs as equally distant if there's no reasonable way to compare them:
        graph_dist = graph_dist or (lambda g1, g2: 1)

        def draw_graph(graph: Graph, ax, title, highlight_title=False):
            ax.clear()
            ax.set_title(title, fontsize=22, color='green' if highlight_title else 'black')
            GraphVisualizer(graph).draw_nx_dag(ax, node_names_placement='legend')

        try:
            last_internal_graph = self.history.archive_history[-1][0]
            genealogical_path = trace_genealogical_path(last_internal_graph, graph_dist)
        except Exception as e:
            # At least `Individual.parents_from_prev_generation` my fail
            self.log.error(f"Failed to trace genealogical path: {e}")
            return

        figure_width = 5
        width_ratios = [1.3, 0.7]
        if target_graph is not None:
            width_ratios = [1.3] + width_ratios

        fig, axes = plt.subplots(
            1, len(width_ratios),
            figsize=(figure_width * sum(width_ratios), figure_width),
            gridspec_kw={'width_ratios': width_ratios}
        )
        evo_ax, fitness_ax = axes[-2:]
        if target_graph is not None:
            draw_graph(target_graph, axes[0], "Target graph")  # Persists throughout the animation

        fitnesses_along_path = list(map(lambda ind: ind.fitness.value, genealogical_path))
        generations_along_path = list(map(lambda ind: ind.native_generation, genealogical_path))

        def render_frame(frame_index):
            path_index = min(frame_index, len(genealogical_path) - 1)
            is_hold_stage = frame_index >= len(genealogical_path)

            draw_graph(
                genealogical_path[path_index].graph, evo_ax,
                f"Evolution process,\ngeneration {generations_along_path[path_index]}/{generations_along_path[-1]}",
                highlight_title=is_hold_stage
            )
            # Select only the genealogical path
            fitness_ax.clear()
            plot_fitness_with_axvline(
                generations=generations_along_path,
                fitnesses=fitnesses_along_path,
                ax=fitness_ax,
                axvline_x=generations_along_path[path_index],
                current_fitness=fitnesses_along_path[path_index]
            )
            return evo_ax, fitness_ax

        frames = len(genealogical_path) + int(
            math.ceil(len(genealogical_path) * hold_result_time_s / (hold_result_time_s + evolution_time_s))
        )
        seconds_per_frame = (evolution_time_s + hold_result_time_s) / frames
        fps = math.ceil(1 / seconds_per_frame)

        anim = animation.FuncAnimation(fig, render_frame, repeat=False, frames=frames,
                                       interval=1000 * seconds_per_frame)

        try:
            if save_path is not None:
                anim.save(save_path, fps=fps)
            if show:
                plt.show()
        except Exception as e:
            self.log.error(f"Failed to render the genealogical path: {e}")


def trace_genealogical_path(individual: Individual, graph_dist: Callable[[Graph, Graph], float]) -> List[Individual]:
    # Choose nearest parent each time:
    genealogical_path: List[Individual] = [individual]
    while genealogical_path[-1].parents_from_prev_generation:
        genealogical_path.append(max(
            genealogical_path[-1].parents_from_prev_generation,
            key=partial(graph_dist, genealogical_path[-1])
        ))

    return list(reversed(genealogical_path))


def plot_fitness_with_axvline(generations: List[int], fitnesses: List[float], ax: plt.Axes, current_fitness: float,
                              axvline_x: int = None):
    ax.plot(generations, fitnesses)
    ax.set_title(f'Fitness dynamic,\ncurrent: {current_fitness}', fontsize=22)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    if axvline_x is not None:
        ax.axvline(x=axvline_x, color='black')
    return ax
