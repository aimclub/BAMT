import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.paths import default_data_dir
from golem.visualisation.opt_history.fitness_line import with_alternate_matplotlib_backend
from golem.visualisation.opt_history.history_visualization import HistoryVisualization
from golem.visualisation.opt_history.utils import show_or_save_figure


class GraphsInteractive(HistoryVisualization):

    @with_alternate_matplotlib_backend
    def visualize(self, save_path: Optional[Union[os.PathLike, str]] = None, dpi: Optional[int] = None,
                  per_time: Optional[bool] = None, graph_show_kwargs: Optional[Dict[str, Any]] = None):
        """ Shows the structure of the best individuals of all time.
        :param save_path: path to save the visualization. If set, then the image will be saved, and if not,
            it will be displayed.
        :param dpi: DPI of the output figure.
        :param per_time: defines whether to show time grid if it is available in history.
        :param graph_show_kwargs: keyword arguments of `graph.show()` function.
        """

        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')
        graph_show_kwargs = graph_show_kwargs or self.visuals_params.get('graph_show_params') or {}

        # fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        # ax_fitness, ax_graph = axes
        fig, ax_graph = plt.subplots(1, 1, figsize=(10, 10))
        ax_graph.axis('off')

        x_template = 'best individual #{}'
        best_individuals = {i: ind
                            for i, ind in enumerate(self.history.archive_history[-1])}

        class InteractivePlot:
            temp_path = Path(default_data_dir(), 'current_graph.png')

            def __init__(self, best_individuals: Dict[int, Individual]):
                self.best_x: List[int] = list(best_individuals.keys())
                self.best_individuals: List[Individual] = list(best_individuals.values())
                self.index: int = len(self.best_individuals) - 1
                # self.time_line = ax_fitness.axvline(self.best_x[self.index], color='r', alpha=0.7)
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

            # def update_time_line(self):
            #     self.time_line.set_xdata(self.best_x[self.index])

            def step_index(self, step: int):
                self.index = (self.index + step) % len(self.best_individuals)
                self.update_graph()
                # self.update_time_line()
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
