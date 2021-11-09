import itertools

from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch
import Nodes
import logging.config

from os import path

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')

logging.config.fileConfig(log_file_path)
logger = logging.getLogger('builder')

from redef_HC import hc as hc_method


class StructureBuilder(object):
    """
    Base Class for Structure Builder.
    It can restrict nodes defined by RESTRICTIONS
    """
    has_logit: bool

    def __init__(self, descriptor):
        """
        :param descriptor: a dict with types and signs of nodes
        Attributes:
        Skeleton;
        black_list: a list with restricted connections;
        white_list: a list with allowed connections.
        """
        self.skeleton = {'V': None,
                         'E': None}
        self.descriptor = descriptor

        self.black_list = None
        self.white_list = None

    def restrict(self, data, init_nodes, cont_disc, bl_add):
        """
        :param data: data to deal with
        :param init_nodes: nodes to begin with
        :param cont_disc: allow continuous-discrete edges
        :param bl_add: additional vertices
        """
        node_type = self.descriptor['types']
        blacklist = []
        datacol = data.columns.to_list()
        if not self.has_logit:
            RESTRICTIONS = [('cont', 'disc'), ('cont', 'disc_num')]
        else:
            # TODO: есть ли тут запреты тогда?
            RESTRICTIONS = []
            self.black_list = []
            return None
        if init_nodes:
            blacklist = [(x, y) for x in datacol for y in init_nodes if x != y]
        if not cont_disc:
            for x, y in itertools.product(datacol, repeat=2):
                if x != y:
                    if (node_type[x], node_type[y]) in RESTRICTIONS:
                        blacklist.append((x, y))
        if bl_add:
            blacklist = blacklist + bl_add
        self.black_list = blacklist

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def get_family(self):
        """
        A function that update a skeleton;
        Represenent the second level of defining vertices according their parents
        :return:
        """
        assert self.skeleton['V'], "Vertex list is None"
        assert self.skeleton['E'], "Edges list is None"
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


class VerticesDefiner(StructureBuilder):
    """
    Main class for defining vertices
    """

    def __init__(self, descriptor):
        """
        Automatically creates a list of nodes
        """
        super(VerticesDefiner, self).__init__(descriptor=descriptor)
        self.vertices = []

        Node = None
        # LEVEL 1: Определение первичной структуры (без учета типа родителей)
        for vertex, type in self.descriptor['types'].items():
            if type in ['disc_num', 'disc']:
                Node = Nodes.DiscreteNode(name=vertex)
            elif type == 'cont':
                Node = Nodes.GaussianNode(name=vertex)
            else:
                msg = f"""First stage of automatic vertex detection failed on "{vertex}" due TypeError ({type}). Set vertex manually (by calling set_nodes()) or investigate the error."""
                logger.error(msg)
                continue

            self.vertices.append(Node)

    # TODO: NAMING
    def define_vertex(self, has_logit, use_mixture):
        # TODO: Нарушение совместимости типов, придумать контсруктор условия
        for node_instance in self.vertices:
            if has_logit:
                # Ideal: if node_instance.type in ['disc_num', 'disc']
                if 'Discrete' in node_instance.type:
                    if node_instance.cont_parents:
                        if not node_instance.disc_parents:
                            Node = Nodes.LogitNode(name=node_instance.name)
                        elif node_instance.disc_parents:
                            Node = Nodes.ConditionalLogitNode(name=node_instance.name)

            if use_mixture:
                #Ideal: if node_instance.type in ['cont']
                if 'Gaussian' in node_instance.type:
                    if not node_instance.disc_parents:
                        Node = Nodes.MixtureGaussianNode(name=node_instance.name)
                    elif node_instance.disc_parents:
                        Node = Nodes.ConditionalMixtureGaussianNode(name=node_instance.name)
                    else:
                        continue
            else:
                if node_instance.disc_parents:
                    Node = Nodes.ConditionalGaussianNode(name=node_instance.name)
                else:
                    continue

            id = self.skeleton['V'].index(node_instance)
            Node.disc_parents = node_instance.disc_parents
            Node.cont_parents = node_instance.cont_parents
            Node.children = node_instance.children
            self.skeleton['V'][id] = Node


class EdgesDefiner(StructureBuilder):
    def __init__(self, descriptor):
        super(EdgesDefiner, self).__init__(descriptor)


class HillClimbDefiner(EdgesDefiner, VerticesDefiner):
    def __init__(self, data, descriptor,
                 scoring_function: tuple):
        """
        :param scoring_function: a tuple with following format (Name, scoring_function)
        """
        if len(scoring_function) == 2:
            assert callable(scoring_function[1]), "Cannot call scoring function"
        self.scoring_function = scoring_function
        self.optimizer = HillClimbSearch(data)
        self.params = {'init_edges': None,
                       'init_nodes': None,
                       'remove_init_edges': False}
        super(HillClimbDefiner, self).__init__(descriptor)

    # TODO: Все доп параметры в kwargs
    def apply_K2(self, data, params=None):
        from Preprocessors import BasePreprocessor
        if not all([i in ['disc', 'disc_num'] for i in BasePreprocessor.get_nodes_types(data).values()]):
            logger.error(
                f"K2 deals only with discrete data. Continuous data: {[col for col, type in BasePreprocessor.get_nodes_types(data).items() if type not in ['disc', 'disc_num']]}")
            return None
        assert self.scoring_function[0] == 'K2'
        scoring_function = self.scoring_function[1]
        # TODO: Можно ли это исправить? Исправить передачу параметров
        if params:
            self.params = params

        init_edges = self.params['init_edges']
        init_nodes = self.params['init_nodes']
        remove_init_edges = self.params['remove_init_edges']

        if not init_edges:
            best_model = self.optimizer.estimate(
                scoring_method=scoring_function(data),
                black_list=self.black_list,
                white_list=self.white_list,
            )
        else:
            if remove_init_edges:
                startdag = DAG()
                startdag.add_nodes_from(nodes=self.vertices)
                startdag.add_edges_from(ebunch=init_edges)
                best_model = self.optimizer.estimate(black_list=self.black_list, white_list=self.white_list,
                                                     start_dag=startdag, show_progress=False)
            else:
                best_model = self.optimizer.estimate(black_list=self.black_list, white_list=self.white_list,
                                                     fixed_edges=init_edges, show_progress=False)

        structure = [list(x) for x in list(best_model.edges())]
        self.skeleton['E'] = structure

    def apply_group1(self, data, init_edges, remove_init_edges):
        # (score == "MI") | (score == "LL") | (score == "BIC") | (score == "AIC")
        column_name_dict = dict([(n.name, i) for i, n in enumerate(self.vertices)])
        blacklist_new = []
        for pair in self.black_list:
            blacklist_new.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
        if self.white_list:
            white_list_old = self.white_list[:]
            white_list = []
            for pair in white_list_old:
                white_list.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
        if init_edges:
            init_edges_old = init_edges[:]
            init_edges = []
            for pair in init_edges_old:
                init_edges.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))

        bn = hc_method(data, metric=self.scoring_function[0], restriction=self.white_list, init_edges=init_edges,
                       remove_geo_edges=remove_init_edges, black_list=blacklist_new, debug=False)
        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            for pa in bn.F[rv]['parents']:
                structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                                  list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
        self.skeleton['E'] = structure


class HCStructureBuilder(HillClimbDefiner, StructureBuilder):
    def __init__(self, data, descriptor, scoring_function, has_logit, use_mixture):
        self.use_mixture = use_mixture
        self.has_logit = has_logit
        super(HCStructureBuilder, self).__init__(descriptor=descriptor, data=data,
                                                 scoring_function=scoring_function)

    def build(self, data, init_nodes, cont_disc, bl_add):
        self.skeleton['V'] = self.vertices
        self.restrict(data, init_nodes, cont_disc, bl_add)
        if self.scoring_function[0] == 'K2':
            self.apply_K2(data=data)
        elif self.scoring_function[0] in ['MI', 'LL', 'BIC', 'AIC']:
            self.apply_group1(data, None, True)

        self.get_family()
        # 2 stage
        if self.has_logit or self.use_mixture:
            self.define_vertex(has_logit=self.has_logit, use_mixture=self.use_mixture)
