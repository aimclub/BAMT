import itertools

from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.gaussian_node import GaussianNode
from bamt.nodes.conditional_logit_node import ConditionalLogitNode
from bamt.nodes.logit_node import LogitNode
from bamt.nodes.mixture_gaussian_node import MixtureGaussianNode
from bamt.nodes.conditional_mixture_gaussian_node import ConditionalMixtureGaussianNode
from bamt.nodes.conditional_gaussian_node import ConditionalGaussianNode

from bamt.log import logger_builder
from pandas import DataFrame
from bamt.utils import GraphUtils as gru

from typing import Dict, List, Optional, Tuple, Callable, TypedDict, Sequence, Union


class ParamDict(TypedDict, total=False):
    init_edges: Optional[Sequence[str]]
    init_nodes: Optional[List[str]]
    remove_init_edges: bool
    white_list: Optional[Tuple[str, str]]
    bl_add: Optional[List[str]]


class StructureBuilder(object):
    """
    Base Class for Structure Builder.
    It can restrict nodes defined by RESTRICTIONS
    """

    def __init__(self, descriptor: Dict[str, Dict[str, str]]):
        """
        :param descriptor: a dict with types and signs of nodes
        Attributes:
        black_list: a list with restricted connections;
        """
        self.skeleton = {'V': [],
                         'E': []}
        self.descriptor = descriptor

        self.has_logit = bool

        self.black_list = None

    def restrict(self, data: DataFrame,
                 init_nodes: Optional[List[str]],
                 bl_add: Optional[List[str]]):
        """
        :param data: data to deal with
        :param init_nodes: nodes to begin with (thus they have no parents)
        :param bl_add: additional vertices
        """
        node_type = self.descriptor['types']
        blacklist = []
        datacol = data.columns.to_list()

        if not self.has_logit:
            # Has_logit flag allows BN building edges between cont and disc
            RESTRICTIONS = [('cont', 'disc'), ('cont', 'disc_num')]
            for x, y in itertools.product(datacol, repeat=2):
                if x != y:
                    if (node_type[x], node_type[y]) in RESTRICTIONS:
                        blacklist.append((x, y))
        else:
            self.black_list = []
        if init_nodes:
            blacklist += [(x, y)
                          for x in datacol for y in init_nodes if x != y]
        if bl_add:
            blacklist = blacklist + bl_add
        self.black_list = blacklist

    def get_family(self):
        """
        A function that updates a skeleton;
        """
        if not self.skeleton['V']:
            logger_builder.error("Vertex list is None")
            return None
        if not self.skeleton['E']:
            logger_builder.error("Edges list is None")
            return None
        for node_instance in self.skeleton['V']:
            node = node_instance.name
            children = []
            parents = []
            for edge in self.skeleton['E']:
                if node in edge:
                    if edge.index(node) == 0:
                        children.append(edge[1])
                    if edge.index(node) == 1:
                        parents.append(edge[0])

            disc_parents = []
            cont_parents = []
            for parent in parents:
                if self.descriptor['types'][parent] in ['disc', 'disc_num']:
                    disc_parents.append(parent)
                else:
                    cont_parents.append(parent)

            id = self.skeleton['V'].index(node_instance)
            self.skeleton['V'][id].disc_parents = disc_parents
            self.skeleton['V'][id].cont_parents = cont_parents
            self.skeleton['V'][id].children = children

        ordered = gru.toporder(self.skeleton['V'], self.skeleton['E'])
        not_ordered = [node.name for node in self.skeleton['V']]
        mask = [not_ordered.index(name) for name in ordered]
        self.skeleton['V'] = [self.skeleton['V'][i] for i in mask]


class VerticesDefiner(StructureBuilder):
    """
    Main class for defining vertices
    """

    def __init__(self, descriptor: Dict[str, Dict[str, str]],
                 regressor: Optional[object]):
        """
        Automatically creates a list of nodes
        """
        super(VerticesDefiner, self).__init__(descriptor=descriptor)
        # Notice that vertices are used only by Builders
        self.vertices = []

        node = None
        # LEVEL 1: Define a general type of node: Discrete or Gaussian
        for vertex, type in self.descriptor['types'].items():
            if type in ['disc_num', 'disc']:
                node = DiscreteNode(name=vertex)
            elif type == 'cont':
                node = GaussianNode(name=vertex, regressor=regressor)
            else:
                msg = f"""First stage of automatic vertex detection failed on {vertex} due TypeError ({type}).
                Set vertex manually (by calling set_nodes()) or investigate the error."""
                logger_builder.error(msg)
                continue

            self.vertices.append(node)

    def overwrite_vertex(
            self,
            has_logit: bool,
            use_mixture: bool,
            classifier: Optional[Callable],
            regressor: Optional[Callable]):
        """
        Level 2: Redefined nodes according structure (parents)
        :param classifier: an object to pass into logit, condLogit nodes
        :param regressor: an object to pass into gaussian nodes
        :param has_logit allows edges from cont to disc nodes
        :param use_mixture allows using Mixture
        """
        for node_instance in self.vertices:
            node = node_instance
            if has_logit:
                if 'Discrete' in node_instance.type:
                    if node_instance.cont_parents:
                        if not node_instance.disc_parents:
                            node = LogitNode(
                                name=node_instance.name, classifier=classifier)

                        elif node_instance.disc_parents:
                            node = ConditionalLogitNode(
                                name=node_instance.name, classifier=classifier)

            if use_mixture:
                if 'Gaussian' in node_instance.type:
                    if not node_instance.disc_parents:
                        node = MixtureGaussianNode(
                            name=node_instance.name)
                    elif node_instance.disc_parents:
                        node = ConditionalMixtureGaussianNode(
                            name=node_instance.name)
                    else:
                        continue
            else:
                if 'Gaussian' in node_instance.type:
                    if node_instance.disc_parents:
                        node = ConditionalGaussianNode(
                            name=node_instance.name, regressor=regressor)
                    else:
                        continue

            if node_instance == node:
                continue

            id = self.skeleton['V'].index(node_instance)
            node.disc_parents = node_instance.disc_parents
            node.cont_parents = node_instance.cont_parents
            node.children = node_instance.children
            self.skeleton['V'][id] = node


class EdgesDefiner(StructureBuilder):
    def __init__(self, descriptor: Dict[str, Dict[str, str]]):
        super(EdgesDefiner, self).__init__(descriptor)


class BaseDefiner(VerticesDefiner, EdgesDefiner):
    def __init__(self, data: DataFrame, descriptor: Dict[str, Dict[str, str]],
                 scoring_function: Union[Tuple[str, Callable], Tuple[str]],
                 regressor: Optional[object] = None):

        self.scoring_function = scoring_function
        self.params = {'init_edges': None,
                       'init_nodes': None,
                       'remove_init_edges': True,
                       'white_list': None,
                       'bl_add': None}
        super().__init__(descriptor, regressor=regressor)
        self.optimizer = None  # will be defined in subclasses
