import itertools
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch
import Nodes


# from redef_HC import hc as hc_method


class StructureBuilder(object):
    def __init__(self, descriptor):
        self.skeleton = {'V': None,
                         'E': None}
        self.descriptor = descriptor

        self.black_list = None
        self.white_list = None

    def restrict(self, data, init_nodes, cont_disc, bl_add):
        node_type = self.descriptor['types']
        blacklist = []
        datacol = data.columns.to_list()
        RESTRICTIONS = [('cont', 'disc'), ('cont', 'disc_num')]
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


class VerticesDefiner(StructureBuilder):
    def __init__(self, descriptor):
        super(VerticesDefiner, self).__init__(descriptor=descriptor)
        self.vertices = {}

        # TODO: не регулируется!
        Node = None
        for vertice, type in self.descriptor['types'].items():
            if type in ['disc_num', 'disc']:
                Node = Nodes.DiscreteNode(name=vertice, type='Discrete')
            elif type == 'cont':
                Node = Nodes.GaussianNode(name=vertice, type='Gaussian')

            self.vertices[vertice] = Node


class EdgesDefiner(StructureBuilder):
    def __init__(self, descriptor):
        super(EdgesDefiner, self).__init__(descriptor)


class HillClimbDefiner(EdgesDefiner, VerticesDefiner):
    def __init__(self, data, descriptor,
                 scoring_function):
        # scoring_function = (Name, scor_func)
        # assert Callable
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
            print(
                [col for col, type in BasePreprocessor.get_nodes_types(data).items()
                 if type not in ['disc', 'disc_num']]
            )
            raise TypeError("K2 deals only with discrete data")
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
    # not worked
    def apply_group1(self, data, init_edges, remove_init_edges):
        # (score == "MI") | (score == "LL") | (score == "BIC") | (score == "AIC")
        column_name_dict = dict([(n, i) for i, n in enumerate(self.vertices)])
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
                       remove_geo_edges=remove_init_edges, black_list=blacklist_new)
        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            for pa in bn.F[rv]['parents']:
                structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                                  list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
        self.skeleton['E'] = structure


class HCStructureBuilder(HillClimbDefiner, StructureBuilder):
    def __init__(self, data, descriptor, scoring_function):
        super(HCStructureBuilder, self).__init__(descriptor=descriptor, data=data,
                                                 scoring_function=scoring_function)

    def build(self, data, init_nodes, cont_disc, bl_add):
        self.skeleton['V'] = self.vertices
        self.restrict(data, init_nodes, cont_disc, bl_add)
        if self.scoring_function[0] == 'K2':
            self.apply_K2(data=data)
        elif self.scoring_function[0] in ['MI', 'LL', 'BIC', 'AIC']:
            self.apply_group1(data, None, True)
