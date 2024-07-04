from __future__ import annotations

import datetime
import os
from copy import deepcopy
from pathlib import Path
from textwrap import wrap
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Sequence, TYPE_CHECKING, Tuple, Union
from uuid import uuid4

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import ArrowStyle
from pyvis.network import Network
from seaborn import color_palette

from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.dag.graph_utils import distance_to_primary_level
from golem.core.log import default_log
from golem.core.paths import default_data_dir

if TYPE_CHECKING:
    from golem.core.dag.graph import Graph
    from golem.core.optimisers.graph import OptGraph
    from golem.core.dag.graph_node import GraphNode

    GraphType = Union[Graph, OptGraph]
    GraphConvertType = Callable[[GraphType], Tuple[nx.DiGraph, Dict[uuid4, GraphNode]]]

PathType = Union[os.PathLike, str]

MatplotlibColorType = Union[str, Sequence[float]]
LabelsColorMapType = Dict[str, MatplotlibColorType]
NodeColorFunctionType = Callable[[Iterable[str]], LabelsColorMapType]
NodeColorType = Union[MatplotlibColorType, LabelsColorMapType, NodeColorFunctionType]


class GraphVisualizer:
    def __init__(self, graph: GraphType, visuals_params: Optional[Dict[str, Any]] = None,
                 to_nx_convert_func: GraphConvertType = graph_structure_as_nx_graph):
        visuals_params = visuals_params or {}
        default_visuals_params = dict(
            engine='matplotlib',
            dpi=100,
            node_color=self._get_colors_by_labels,
            node_size_scale=1.0,
            font_size_scale=1.0,
            edge_curvature_scale=1.0,
            node_names_placement='auto',
            nodes_layout_function=GraphVisualizer._get_hierarchy_pos_by_distance_to_primary_level,
            figure_size=(7, 7),
            save_path=None,
        )
        default_visuals_params.update(visuals_params)
        self.visuals_params = default_visuals_params
        self.to_nx_convert_func = to_nx_convert_func
        self._update_graph(graph)
        self.log = default_log(self)

    def _update_graph(self, graph: GraphType):
        self.graph = graph
        self.nx_graph, self.nodes_dict = self.to_nx_convert_func(self.graph)

    def visualise(self, save_path: Optional[PathType] = None, engine: Optional[str] = None,
                  node_color: Optional[NodeColorType] = None, dpi: Optional[int] = None,
                  node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
                  edge_curvature_scale: Optional[float] = None, figure_size: Optional[Tuple[int, int]] = None,
                  nodes_labels: Dict[int, str] = None, edges_labels: Dict[int, str] = None,
                  node_names_placement: Optional[Literal['auto', 'nodes', 'legend', 'none']] = None,
                  nodes_layout_function: Optional[Callable[[nx.DiGraph], Dict[Any, Tuple[float, float]]]] = None,
                  title: Optional[str] = None):
        engine = engine or self._get_predefined_value('engine')

        if not self.graph.nodes:
            raise ValueError('Empty graph can not be visualized.')

        if engine == 'matplotlib':
            self._draw_with_networkx(save_path=save_path, node_color=node_color, dpi=dpi,
                                     node_size_scale=node_size_scale, font_size_scale=font_size_scale,
                                     edge_curvature_scale=edge_curvature_scale, figure_size=figure_size,
                                     title=title, nodes_labels=nodes_labels, edges_labels=edges_labels,
                                     nodes_layout_function=nodes_layout_function,
                                     node_names_placement=node_names_placement)
        elif engine == 'pyvis':
            self._draw_with_pyvis(save_path, node_color)
        elif engine == 'graphviz':
            self._draw_with_graphviz(save_path, node_color, dpi)
        else:
            raise NotImplementedError(f'Unexpected visualization engine: {engine}. '
                                      'Possible values: matplotlib, pyvis, graphviz.')

    def draw_nx_dag(
            self, ax: Optional[plt.Axes] = None, node_color: Optional[NodeColorType] = None,
            node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
            edge_curvature_scale: Optional[float] = None, nodes_labels: Dict[int, str] = None,
            edges_labels: Dict[int, str] = None,
            nodes_layout_function: Optional[Callable[[nx.DiGraph], Dict[Any, Tuple[float, float]]]] = None,
            node_names_placement: Optional[Literal['auto', 'nodes', 'legend', 'none']] = None):
        node_color = node_color or self._get_predefined_value('node_color')
        node_size_scale = node_size_scale or self._get_predefined_value('node_size_scale')
        font_size_scale = font_size_scale or self._get_predefined_value('font_size_scale')
        edge_curvature_scale = (edge_curvature_scale if edge_curvature_scale is not None
                                else self._get_predefined_value('edge_curvature_scale'))
        nodes_layout_function = nodes_layout_function or self._get_predefined_value('nodes_layout_function')
        node_names_placement = node_names_placement or self._get_predefined_value('node_names_placement')

        nx_graph, nodes = self.nx_graph, self.nodes_dict

        if ax is None:
            ax = plt.gca()

        # Define colors
        if callable(node_color):
            node_color = node_color([str(node) for node in nodes.values()])
        if isinstance(node_color, dict):
            node_color = [node_color.get(str(node), node_color.get(None)) for node in nodes.values()]
        else:
            node_color = [node_color for _ in nodes]
        # Get node positions
        if nodes_layout_function == GraphVisualizer._get_hierarchy_pos_by_distance_to_primary_level:
            pos = nodes_layout_function(nx_graph, nodes)
        else:
            pos = nodes_layout_function(nx_graph)

        node_size = self._get_scaled_node_size(len(nodes), node_size_scale)

        with_node_names = node_names_placement != 'none'

        if node_names_placement in ('auto', 'none'):
            node_names_placement = GraphVisualizer._define_node_names_placement(node_size)

        if node_names_placement == 'nodes':
            self._draw_nx_big_nodes(ax, pos, nodes, node_color, node_size, font_size_scale, with_node_names)
        elif node_names_placement == 'legend':
            self._draw_nx_small_nodes(ax, pos, nodes, node_color, node_size, font_size_scale, with_node_names)
        self._draw_nx_curved_edges(ax, pos, node_size, edge_curvature_scale)
        self._draw_nx_labels(ax, pos, font_size_scale, nodes_labels, edges_labels)

    def _get_predefined_value(self, param: str):
        if param not in self.visuals_params:
            self.log.warning(f'No default param found: {param}.')
        return self.visuals_params.get(param)

    def _draw_with_networkx(
            self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None,
            dpi: Optional[int] = None, node_size_scale: Optional[float] = None,
            font_size_scale: Optional[float] = None, edge_curvature_scale: Optional[float] = None,
            figure_size: Optional[Tuple[int, int]] = None, title: Optional[str] = None,
            nodes_labels: Dict[int, str] = None, edges_labels: Dict[int, str] = None,
            nodes_layout_function: Optional[Callable[[nx.DiGraph], Dict[Any, Tuple[float, float]]]] = None,
            node_names_placement: Optional[Literal['auto', 'nodes', 'legend', 'none']] = None):
        save_path = save_path or self._get_predefined_value('save_path')
        node_color = node_color or self._get_predefined_value('node_color')
        dpi = dpi or self._get_predefined_value('dpi')
        figure_size = figure_size or self._get_predefined_value('figure_size')

        ax = GraphVisualizer._setup_matplotlib_figure(figure_size, dpi, title)
        self.draw_nx_dag(ax, node_color, node_size_scale, font_size_scale, edge_curvature_scale,
                         nodes_labels, edges_labels, nodes_layout_function, node_names_placement)
        GraphVisualizer._rescale_matplotlib_figure(ax)
        if not save_path:
            plt.show()
        else:
            plt.savefig(save_path, dpi=dpi)
            plt.close()

    def _draw_with_pyvis(self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None):
        save_path = save_path or self._get_predefined_value('save_path')
        node_color = node_color or self._get_predefined_value('node_color')

        net = Network('500px', '1000px', directed=True)
        nx_graph, nodes = self.nx_graph, self.nodes_dict
        node_color = self._define_colors(node_color, nodes)
        for n, data in nx_graph.nodes(data=True):
            operation = nodes[n]
            label = str(operation)
            data['n_id'] = str(n)
            data['label'] = label.replace('_', ' ')
            params = operation.content.get('params')
            if isinstance(params, dict):
                params = str(params)[1:-1]
            data['title'] = params
            data['level'] = distance_to_primary_level(operation)
            data['color'] = to_hex(node_color.get(label, node_color.get(None)))
            data['font'] = '20px'
            data['labelHighlightBold'] = True

        for _, data in nx_graph.nodes(data=True):
            net.add_node(**data)
        for u, v in nx_graph.edges:
            net.add_edge(str(u), str(v))

        if save_path:
            net.save_graph(str(save_path))
            return
        save_path = Path(default_data_dir(), 'graph_plots', str(uuid4()) + '.html')
        save_path.parent.mkdir(exist_ok=True)
        net.show(str(save_path))
        remove_old_files_from_dir(save_path.parent)

    def _draw_with_graphviz(self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None,
                            dpi: Optional[int] = None):
        save_path = save_path or self._get_predefined_value('save_path')
        node_color = node_color or self._get_predefined_value('node_color')
        dpi = dpi or self._get_predefined_value('dpi')

        nx_graph, nodes = self.nx_graph, self.nodes_dict
        node_color = self._define_colors(node_color, nodes)
        for n, data in nx_graph.nodes(data=True):
            label = str(nodes[n])
            data['label'] = label.replace('_', ' ')
            data['color'] = to_hex(node_color.get(label, node_color.get(None)))

        gv_graph = nx.nx_agraph.to_agraph(nx_graph)
        kwargs = {'prog': 'dot', 'args': f'-Gnodesep=0.5 -Gdpi={dpi} -Grankdir="LR"'}

        if save_path:
            gv_graph.draw(save_path, **kwargs)
        else:
            save_path = Path(default_data_dir(), 'graph_plots', str(uuid4()) + '.png')
            save_path.parent.mkdir(exist_ok=True)
            gv_graph.draw(save_path, **kwargs)

            img = plt.imread(str(save_path))
            plt.imshow(img)
            plt.gca().axis('off')
            plt.gcf().set_dpi(dpi)
            plt.tight_layout()
            plt.show()
            remove_old_files_from_dir(save_path.parent)

    @staticmethod
    def _get_scaled_node_size(nodes_amount: int, size_scale: float) -> float:
        min_size = 150
        max_size = 12000
        size = max(max_size * (1 - np.log10(nodes_amount)), min_size)
        return size * size_scale

    @staticmethod
    def _get_scaled_font_size(nodes_amount: int, size_scale: float) -> float:
        min_size = 14
        max_size = 30
        size = max(max_size * (1 - np.log10(nodes_amount)), min_size)
        return size * size_scale

    @staticmethod
    def _get_colors_by_labels(labels: Iterable[str]) -> LabelsColorMapType:
        unique_labels = list(set(labels))
        palette = color_palette('tab10', len(unique_labels))
        return {label: palette[unique_labels.index(label)] for label in labels}

    @staticmethod
    def _define_colors(node_color, nodes):
        if callable(node_color):
            colors = node_color([str(node) for node in nodes.values()])
        elif isinstance(node_color, dict):
            colors = node_color
        else:
            colors = {str(node): node_color for node in nodes.values()}
        return colors

    @staticmethod
    def _setup_matplotlib_figure(figure_size: Tuple[float, float], dpi: int, title: Optional[str] = None) -> plt.Axes:
        fig, ax = plt.subplots(figsize=figure_size)
        fig.set_dpi(dpi)
        plt.title(title)
        return ax

    @staticmethod
    def _rescale_matplotlib_figure(ax):
        """Rescale the figure for all nodes to fit in."""

        x_1, x_2 = ax.get_xlim()
        y_1, y_2 = ax.get_ylim()
        offset = 0.2
        x_offset = x_2 * offset
        y_offset = y_2 * offset
        ax.set_xlim(x_1 - x_offset, x_2 + x_offset)
        ax.set_ylim(y_1 - y_offset, y_2 + y_offset)
        ax.axis('off')
        plt.tight_layout()

    def _draw_nx_big_nodes(self, ax, pos, nodes, node_color, node_size, font_size_scale, with_node_names):
        # Draw the graph's nodes.
        nx.draw_networkx_nodes(self.nx_graph, pos, node_size=node_size, ax=ax, node_color='w', linewidths=3,
                               edgecolors=node_color)
        if not with_node_names:
            return
        # Draw the graph's node labels.
        node_labels = {node_id: str(node) for node_id, node in nodes.items()}
        font_size = GraphVisualizer._get_scaled_font_size(len(nodes), font_size_scale)
        for node, (x, y) in pos.items():
            text = '\n'.join(wrap(node_labels[node].replace('_', ' ').replace('-', ' '), 10))
            ax.text(x, y, text,
                    ha='center', va='center',
                    fontsize=font_size,
                    bbox=dict(alpha=0.9, color='w', boxstyle='round'))

    def _draw_nx_small_nodes(self, ax, pos, nodes, node_color, node_size, font_size_scale, with_node_names):
        nx_graph = self.nx_graph
        markers = 'os^>v<dph8'
        label_markers = {}
        labels_added = set()
        color_counts = {}
        for node_num, (node_id, node) in enumerate(nodes.items()):
            label = str(node)
            color = node_color[node_num]
            color = to_hex(color, keep_alpha=True)  # Convert the color to a hashable type.
            marker = label_markers.get(label)
            if marker is None:
                color_count = color_counts.get(color, 0)
                if color_count > len(markers) - 1:
                    self.log.warning(f'Too much node labels derive the same color: {color}. The markers may repeat.\n'
                                     '\tSpecify the parameter "node_color" to set distinct colors.')
                    color_count = color_count % len(markers)
                marker = markers[color_count]
                label_markers[label] = marker
                color_counts[color] = color_count + 1
            nx.draw_networkx_nodes(nx_graph, pos, [node_id], ax=ax, node_color=[color], node_size=node_size,
                                   node_shape=marker)
            if label in labels_added:
                continue
            ax.plot([], [], marker=marker, linestyle='None', color=color, label=label)
            labels_added.add(label)
        if not with_node_names:
            return
        # @morrisnein took the following code from https://stackoverflow.com/a/27512450
        handles, labels = ax.get_legend_handles_labels()
        # Sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, prop={'size': round(20 * font_size_scale)})

    def _draw_nx_curved_edges(self, ax, pos, node_size, edge_curvature_scale):
        nx_graph = self.nx_graph
        # The ongoing section defines curvature for all edges.
        #   This is 'connection style' for an edge that does not intersect any nodes.
        connection_style = 'arc3'
        #   This is 'connection style' template for an edge that is too close to any node and must bend around it.
        #   The curvature value is defined individually for each edge.
        connection_style_curved_template = connection_style + ',rad={}'
        default_edge_curvature = 0.3
        #   The minimum distance from a node to an edge on which the edge must bend around the node.
        node_distance_gap = 0.15
        for u, v, e in nx_graph.edges(data=True):
            e['connectionstyle'] = connection_style
            p_1, p_2 = np.array(pos[u]), np.array(pos[v])
            p_1_2 = p_2 - p_1
            p_1_2_length = np.linalg.norm(p_1_2)
            # Finding the closest node to the edge.
            min_distance_found = node_distance_gap * 2  # It just must be bigger than the gap.
            closest_node_id = None
            for node_id in nx_graph.nodes:
                if node_id in (u, v):
                    continue  # The node is adjacent to the edge.
                p_3 = np.array(pos[node_id])
                distance_to_node = abs(np.cross(p_1_2, p_3 - p_1)) / p_1_2_length
                if (distance_to_node > min(node_distance_gap, min_distance_found)  # The node is too far.
                        or ((p_3 - p_1) @ p_1_2) < 0  # There's no perpendicular from the node to the edge.
                        or ((p_3 - p_2) @ -p_1_2) < 0):
                    continue
                min_distance_found = distance_to_node
                closest_node_id = node_id

            if closest_node_id is None:
                continue  # There's no node to bend around.
            # Finally, define the edge's curvature based on the closest node position.
            p_3 = np.array(pos[closest_node_id])
            p_1_3 = p_3 - p_1
            curvature_strength = default_edge_curvature * edge_curvature_scale
            # 'alpha' denotes the angle between the abscissa and the edge.
            cos_alpha = p_1_2[0] / p_1_2_length
            sin_alpha = np.sqrt(1 - cos_alpha ** 2) * (-1) ** (p_1_2[1] < 0)
            # The closest node is placed as if the edge matched the abscissa.
            # Then, its ordinate shows on which side of the edge it is, "on the left" or "on the right".
            rotation_matrix = np.array([[cos_alpha, sin_alpha], [-sin_alpha, cos_alpha]])
            p_1_3_rotated = rotation_matrix @ p_1_3
            curvature_direction = (-1) ** (p_1_3_rotated[1] < 0)  # +1 is a "cup" \/, -1 is a "cat" /\.
            edge_curvature = curvature_direction * curvature_strength
            e['connectionstyle'] = connection_style_curved_template.format(edge_curvature)
            # Define edge center position for labels.
            edge_center_position = np.mean([p_1, p_2], axis=0)
            edge_curvature_shift = np.linalg.inv(rotation_matrix) @ [0, -1 * edge_curvature / 4]
            edge_center_position += edge_curvature_shift
            e['edge_center_position'] = edge_center_position
        # Draw the graph's edges.
        arrow_style = ArrowStyle('Simple', head_length=1.5, head_width=0.8)
        for u, v, e in nx_graph.edges(data=True):
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], node_size=node_size, ax=ax, arrowsize=10,
                                   arrowstyle=arrow_style, connectionstyle=e['connectionstyle'])
        self._rescale_matplotlib_figure(ax)

    def _draw_nx_labels(self, ax: plt.Axes, pos: Any, font_size_scale: float,
                        nodes_labels: Dict[int, str], edges_labels: Dict[int, str]):
        """ Set labels with scores to nodes and edges. """

        def calculate_labels_bias(ax: plt.Axes, y_span: int):
            y_1, y_2 = ax.get_ylim()
            y_size = y_2 - y_1
            if y_span == 1:
                bias_scale = 0.25  # Fits between the central line and the upper bound.
            else:
                bias_scale = 1 / y_span / 3 * 0.5  # Fits between the narrowest horizontal rows.
            bias = y_size * bias_scale
            return bias

        def match_labels_with_nx_nodes(nx_graph: nx.DiGraph, labels: Dict[int, str]) -> Dict[str, str]:
            """ Matches index of node in GOLEM graph with networkx node name. """
            nx_nodes = list(nx_graph.nodes.keys())
            nx_labels = {}
            for index, label in labels.items():
                nx_labels[nx_nodes[index]] = label
            return nx_labels

        def match_labels_with_nx_edges(nx_graph: nx.DiGraph, labels: Dict[int, str]) \
                -> Dict[Tuple[str, str], str]:
            """ Matches index of edge in GOLEM graph with tuple of networkx nodes names. """
            nx_nodes = list(nx_graph.nodes.keys())
            edges = self.graph.get_edges()
            nx_labels = {}
            for index, label in labels.items():
                edge = edges[index]
                parent_node_nx = nx_nodes[self.graph.nodes.index(edge[0])]
                child_node_nx = nx_nodes[self.graph.nodes.index(edge[1])]
                nx_labels[(parent_node_nx, child_node_nx)] = label
            return nx_labels

        def draw_node_labels(node_labels, ax, bias, font_size, nx_graph, pos):
            labels_pos = deepcopy(pos)
            for value in labels_pos.values():
                value[1] += bias
            bbox = dict(alpha=0.9, color='w')

            nodes_nx_labels = match_labels_with_nx_nodes(nx_graph=nx_graph, labels=node_labels)
            nx.draw_networkx_labels(
                nx_graph, labels_pos,
                ax=ax,
                labels=nodes_nx_labels,
                font_color='black',
                font_size=font_size,
                bbox=bbox
            )

        def draw_edge_labels(edge_labels, ax, bias, font_size, nx_graph, pos):
            labels_pos_edges = deepcopy(pos)
            label_bias_y = 2 / 3 * bias
            if len(set([coord[1] for coord in pos.values()])) == 1 and len(list(pos.values())) > 2:
                for value in labels_pos_edges.values():
                    value[1] += label_bias_y
            edges_nx_labels = match_labels_with_nx_edges(nx_graph=nx_graph, labels=edge_labels)
            bbox = dict(alpha=0.9, color='w')
            # Set labels for edges
            for u, v, e in nx_graph.edges(data=True):
                if (u, v) not in edges_nx_labels:
                    continue
                current_pos = labels_pos_edges
                if 'edge_center_position' in e:
                    x, y = e['edge_center_position']
                    plt.text(x, y, edges_nx_labels[(u, v)], bbox=bbox, fontsize=font_size)
                else:
                    nx.draw_networkx_edge_labels(
                        nx_graph, current_pos, {(u, v): edges_nx_labels[(u, v)]},
                        label_pos=0.5, ax=ax,
                        font_color='black',
                        font_size=font_size,
                        rotate=False,
                        bbox=bbox
                    )

        if not (edges_labels or nodes_labels):
            return

        nodes_amount = len(pos)
        font_size = GraphVisualizer._get_scaled_font_size(nodes_amount, font_size_scale * 0.75)
        _, y_span = GraphVisualizer._get_x_y_span(pos)
        bias = calculate_labels_bias(ax, y_span)

        if nodes_labels:
            draw_node_labels(nodes_labels, ax, bias, font_size, self.nx_graph, pos)

        if edges_labels:
            draw_edge_labels(edges_labels, ax, bias, font_size, self.nx_graph, pos)

    @staticmethod
    def _get_hierarchy_pos_by_distance_to_primary_level(nx_graph: nx.DiGraph, nodes: Dict
                                                        ) -> Dict[Any, Tuple[float, float]]:
        """By default, returns 'networkx.multipartite_layout' positions based on 'hierarchy_level`
        from node data - the property must be set beforehand.
        :param graph: the graph.
        """
        for node_id, node_data in nx_graph.nodes(data=True):
            node_data['hierarchy_level'] = distance_to_primary_level(nodes[node_id])

        return nx.multipartite_layout(nx_graph, subset_key='hierarchy_level')

    @staticmethod
    def _get_x_y_span(pos: Dict[Any, Tuple[float, float]]) -> Tuple[int, int]:
        pos_x, pos_y = np.split(np.array(tuple(pos.values())), 2, axis=1)
        x_span = max(pos_x) - min(pos_x)
        y_span = max(pos_y) - min(pos_y)
        return x_span, y_span

    @staticmethod
    def _define_node_names_placement(node_size):
        if node_size >= 1000:
            node_names_placement = 'nodes'
        else:
            node_names_placement = 'legend'
        return node_names_placement


def remove_old_files_from_dir(dir_: Path, time_interval=datetime.timedelta(minutes=10)):
    for path in dir_.iterdir():
        if datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_ctime) > time_interval:
            path.unlink()
