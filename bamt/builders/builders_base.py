import itertools
from typing import Dict, List, Optional, Tuple, Callable, Union

from pandas import DataFrame

from bamt.log import logger_builder
from bamt.nodes.conditional_gaussian_node import ConditionalGaussianNode
from bamt.nodes.conditional_logit_node import ConditionalLogitNode
from bamt.nodes.conditional_mixture_gaussian_node import ConditionalMixtureGaussianNode
from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.gaussian_node import GaussianNode
from bamt.nodes.logit_node import LogitNode
from bamt.nodes.mixture_gaussian_node import MixtureGaussianNode
from bamt.utils import graph_utils as gru

from bamt.checkers.enums import continuous_nodes


class StructureBuilder(object):
    """
    Base Class for Structure Builder.
    It can restrict nodes defined by RESTRICTIONS
    """

    def __init__(self, descriptor: Dict[str, Dict[str, str]], **kwargs):
        """
        :param descriptor: a dict with types and signs of nodes
        Attributes:
        black_list: a list with restricted connections;
        """
        # name "nodes" in builders' space is forbidden!
        self.vertices = []
        # name "edges" in builders' space is forbidden!
        self.structure = []
        self.descriptor = descriptor

        self.has_logit = bool

        self.black_list = None

        self.checker_descriptor = kwargs["checkers_rules"]["descriptor"]
        self.is_restricted_pair = kwargs["checkers_rules"]["restriction_rule"]

    def restrict(
        self,
        data: DataFrame,
        init_nodes: Optional[List[str]],
        bl_add: Optional[List[str]],
    ):
        """
        :param data: data to deal with
        :param init_nodes: nodes to begin with (thus they have no parents)
        :param bl_add: additional vertices
        """
        blacklist = []
        datacol = data.columns.to_list()

        if not self.has_logit:
            # Has_logit flag allows BN building edges between cont and disc
            for x, y in itertools.product(datacol, repeat=2):
                if x != y:
                    if self.is_restricted_pair(x, y):
                        blacklist.append((x, y))
        else:
            self.black_list = []
        if init_nodes:
            blacklist += [(x, y) for x in datacol for y in init_nodes if x != y]
        if bl_add:
            blacklist = blacklist + bl_add
        self.black_list = blacklist

    def get_family(self):
        """
        A function that updates each node accordingly structure;
        """
        if not self.vertices:
            logger_builder.error("Vertex list is None")
            return None
        if not self.structure:
            logger_builder.error("Edges list is None")
            return None

        node_mapping = {
            node_instance.name: {"disc_parents": [], "cont_parents": [], "children": []}
            for node_instance in self.vertices
        }

        for edge in self.structure:
            parent, child = edge[0], edge[1]
            node_mapping[parent]["children"].append(child)
            if self.checker_descriptor["types"][parent].is_cont:
                node_mapping[child]["cont_parents"].append(parent)
            else:
                node_mapping[child]["disc_parents"].append(parent)

        for node, data in node_mapping.items():
            node_instance = next(n for n in self.vertices if n.name == node)
            node_instance.disc_parents = data["disc_parents"]
            node_instance.cont_parents = data["cont_parents"]
            node_instance.children = data["children"]

        ordered = gru.toporder(self.vertices, self.structure)
        not_ordered = [node.name for node in self.vertices]
        mask = [not_ordered.index(name) for name in ordered]
        self.vertices = [self.vertices[i] for i in mask]


class VerticesDefiner(StructureBuilder):
    """
    Main class for defining vertices
    """

    def __init__(
        self,
        descriptor: Dict[str, Dict[str, str]],
        regressor: Optional[object],
        **kwargs,
    ):
        """
        Automatically creates a list of nodes
        """
        super().__init__(descriptor=descriptor, **kwargs)
        self.regressor = regressor

    def init_nodes(self):
        vertices = []
        node = None
        # LEVEL 1: Define a general type of node: Discrete or Gaussian
        for vertex, vertex_checker in self.checker_descriptor["types"].items():
            if not vertex_checker.is_cont:
                node = DiscreteNode(name=vertex)
            elif vertex_checker.is_cont:
                node = GaussianNode(name=vertex, regressor=self.regressor)
            else:
                msg = f"""First stage of automatic vertex detection failed on {vertex} due TypeError ({type}).
                        Set vertex manually (by calling set_nodes()) or investigate the error."""
                logger_builder.error(msg)

            vertices.append(node)
        return vertices


class EdgesDefiner(StructureBuilder):
    def __init__(self, vertices, descriptor: Dict[str, Dict[str, str]], **kwargs):
        super().__init__(descriptor, **kwargs)
        self.vertices = vertices
        self.structure = []

    def overwrite_vertex(
        self,
        has_logit: bool,
        use_mixture: bool,
        classifier: Optional[Callable],
        regressor: Optional[Callable],
    ):
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
                if node_instance.cont_parents:
                    if not node_instance.disc_parents:
                        node = LogitNode(name=node_instance.name, classifier=classifier)

                    elif node_instance.disc_parents:
                        node = ConditionalLogitNode(
                            name=node_instance.name, classifier=classifier
                        )

            if node_instance.type in continuous_nodes:
                if use_mixture:
                    if not node_instance.disc_parents:
                        node = MixtureGaussianNode(name=node_instance.name)
                    elif node_instance.disc_parents:
                        node = ConditionalMixtureGaussianNode(name=node_instance.name)
                    else:
                        continue
                else:
                    if node_instance.disc_parents:
                        node = ConditionalGaussianNode(
                            name=node_instance.name, regressor=regressor
                        )
                    else:
                        # redefine default node with new regressor
                        node = GaussianNode(
                            name=node_instance.name, regressor=regressor
                        )

            node_checker = self.checker_descriptor["types"][node.name].evolve(
                node.type, node_instance.cont_parents, node_instance.disc_parents
            )

            self.checker_descriptor["types"][node.name] = node_checker

            if node_instance == node:
                continue

            id = self.vertices.index(node_instance)
            node.disc_parents = node_instance.disc_parents
            node.cont_parents = node_instance.cont_parents
            node.children = node_instance.children
            self.vertices[id] = node


class BaseDefiner(EdgesDefiner, VerticesDefiner):
    def __init__(
        self,
        data: DataFrame,
        vertices: list,
        descriptor: Dict[str, Dict[str, str]],
        scoring_function: Union[Tuple[str, Callable], Tuple[str]],
        checkers_rules: dict,
    ):
        self.scoring_function = scoring_function
        self.params = {
            "init_edges": None,
            "init_nodes": None,
            "remove_init_edges": True,
            "white_list": None,
            "bl_add": None,
        }

        super().__init__(
            descriptor=descriptor,
            vertices=vertices,
            checkers_rules=checkers_rules,
            regressor=None,
        )

        self.optimizer = None  # will be defined in subclasses
