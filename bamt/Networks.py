from bamt import Builders, Nodes
import random
import re
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from bamt.utils import GraphUtils as gru
from pyvis.network import Network

from typing import Dict, Tuple, List, Callable, Optional, Type, Union
from bamt.Builders import ParamDict

from bamt.log import logger_network
from bamt.config import config

STORAGE = config.get('NODES', 'models_storage', fallback='models_storage is not defined')

class BaseNetwork(object):
    """
    Base class for Bayesian Network
    """

    def __init__(self):
        """
        Attributes:
            nodes: a list of nodes instances
            edges: a list of edges
            distributions: dict
        """
        self.type = 'Abstract'
        self._allowed_dtypes = ['Abstract']
        self.nodes = []
        self.edges = []
        self.descriptor = {}
        self.distributions = {}
        self.has_logit = False
        self.use_mixture = False


    @property
    def nodes_names(self) -> List[str]:
        return [node.name for node in self.nodes]

    def __getitem__(self, node_name: str) -> Type[Nodes.BaseNode]:
        index = self.nodes_names.index(node_name)
        return self.nodes[index]

    def validate(self, descriptor: Dict[str, Dict[str, str]]) -> bool:
        types = descriptor['types']
        return True if all([a in self._allowed_dtypes for a in types.values()]) else False

    def update_descriptor(self):
        new_nodes_names = [node.name for node in self.nodes]
        self.descriptor['types'] = {node: type for node, type in self.descriptor['types'].items() if
                                    node in new_nodes_names}
        self.descriptor['signs'] = {node: sign for node, sign in self.descriptor['signs'].items() if
                                    node in new_nodes_names}

    def add_nodes(self, descriptor: Dict[str, Dict[str, str]]):
        """
        Function for initializing nodes in Bayesian Network
        descriptor: dict with types and signs of nodes
        """
        self.descriptor = descriptor
        if not self.validate(descriptor=descriptor):
            if not self.type == 'Hybrid':
                logger_network.error(
                    f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
                return
            else:
                logger_network.error(
                    f"Hybrid BN is not supposed to work with only one type of data. Use DiscreteBN or Continuous BN instead.")
                return
        elif ['Abstract'] in self._allowed_dtypes:
            return None
        # LEVEL 1
        worker_1 = Builders.VerticesDefiner(descriptor)
        self.nodes = worker_1.vertices

    def add_edges(self, data: pd.DataFrame, scoring_function: Union[Tuple[str, Callable], Tuple[str]],
                  classifier: Optional[object] = None,
                  params: Optional[ParamDict] = None, optimizer: str = 'HC'):
        """
        Base function for Structure learning
        scoring_function: tuple with following format (NAME, scoring_function) or (NAME,)
        Params:
        init_edges: list of tuples, a graph to start learning with
        remove_init_edges: allows changes in model defined by user
        white_list: list of allowed edges
        """
        if not self.has_logit and classifier:
            logger_network.error("Classifiers dict will be ignored since logit nodes are forbidden.")
            return None

        if not self.validate(descriptor=self.descriptor):
            logger_network.error(
                f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
            return None
        if optimizer == 'HC':
            worker = Builders.HCStructureBuilder(data=data,
                                                 descriptor=self.descriptor,
                                                 scoring_function=scoring_function,
                                                 has_logit=self.has_logit,
                                                 use_mixture=self.use_mixture)
            self.sf_name = scoring_function[0]
            worker.build(data=data, params=params, classifier=classifier)

            # update family
            self.nodes = worker.skeleton['V']
            self.edges = worker.skeleton['E']

    def set_nodes(self, nodes: Optional[Dict[str, Type[Nodes.BaseNode]]] = None, **kwargs):
        """
        additional function to set nodes manually. User should be aware that
        nodes must be a subclass of BaseNode.
        :param nodes dict with name and node (if a lot of nodes should be added)
        """
        if nodes is None:
            nodes = dict()
        nodes.update(kwargs)
        for column_name, node in nodes.items():
            try:
                assert issubclass(node, Nodes.BaseNode)
            except AssertionError:
                logger_network.error(f"{node} is not an instance of {Nodes.BaseNode}")
                continue
            except TypeError:
                logger_network.error(f"Passed kwarg must be a class. Arg: {node}")
                continue

            self.nodes.append(node(name=column_name))
            self.update_descriptor()

    def get_params_tree(self, outdir: str):
        """
        Function to save BN params to json file
        outdir: output directory
        """
        if not outdir.endswith('.json'):
            return None
        with open(outdir, 'w+') as out:
            json.dump(self.distributions, out)
        return True

    def fit_parameters(self, data: pd.DataFrame, dropna: bool = True):
        """
        Base function for parameters learning
        """
        if dropna:
            data = data.dropna()
            data.reset_index(inplace=True, drop=True)

        if self.has_logit:
            if any(['Logit' in node.type for node in self.nodes]):
                if not os.path.isdir(STORAGE):
                    os.makedirs(os.path.join(STORAGE, "0"))
                elif os.listdir(STORAGE):
                    index = sorted(
                    [int(id) for id in os.listdir(STORAGE)]
                    )[-1] + 1
                    os.makedirs(os.path.join(STORAGE, str(index)))

        # Turn all discrete values to str for learning algorithm
        if 'disc_num' in self.descriptor['types'].values():
            columns_names = [name for name, t in self.descriptor['types'].items() if t in ['disc_num']]
            data[columns_names] = data.loc[:, columns_names].astype('str')

        # Topology sorting
        ordered = gru.toporder(self.nodes, self.edges)
        notOrdered = [node.name for node in self.nodes]
        mask = [notOrdered.index(name) for name in ordered]
        self.nodes = [self.nodes[i] for i in mask]

        def worker(node):
            return node.fit_parameters(data)

        pool = ThreadPoolExecutor(3)
        for node in self.nodes:
            future = pool.submit(worker, node)
            self.distributions[node.name] = future.result()

    def get_info(self, as_df: bool = True) -> Optional[pd.DataFrame]:
        """Return a table with name, type, parents_type, parents_names"""
        if as_df:
            names = []
            types_n = []
            types_d = []
            parents = []
            parents_types = []
            for n in self.nodes:
                names.append(n)
                types_n.append(n.type)
                types_d.append(self.descriptor['types'][n.name])
                parents_types.append([self.descriptor['types'][name] for name in n.cont_parents + n.disc_parents])
                parents.append([name for name in n.cont_parents + n.disc_parents])
            return pd.DataFrame({'name': names, 'node_type': types_n,
                                 'data_type': types_d, 'parents': parents,
                                 'parents_types': parents_types})
        else:
            for n in self.nodes:
                print(
                    f"{n.name: <20} | {n.type: <50} | {self.descriptor['types'][n.name]: <10} | {str([self.descriptor['types'][name] for name in n.cont_parents + n.disc_parents]): <50} | {str([name for name in n.cont_parents + n.disc_parents])}")

    def sample(self, n: int, evidence: Optional[Dict[str, Union[str, int, float]]] = None,
               as_df: bool = True) -> Union[None, pd.DataFrame, List[Dict[str, Union[str, int, float]]]]:
        """
        Sampling from Bayesian Network
        n: int number of samples
        evidence: values for nodes from user
        """
        seq = []
        random.seed()
        if not self.distributions.items():
            logger_network.error("Parameter learning wasn't done. Call fit_parameters method")
            return None
        for n in range(n):
            output = {}
            for node in self.nodes:
                parents = node.cont_parents + node.disc_parents
                if evidence:
                    if node.name in evidence.keys():
                        output[node.name] = evidence[node.name]
                    else:
                        if not parents:
                            pvals = None
                        else:
                            if self.type == 'Discrete':
                                pvals = [str(output[t]) for t in parents]
                            else:
                                pvals = [output[t] for t in parents]
                        output[node.name] = node.choose(self.distributions[node.name], pvals=pvals)
                else:
                    if not parents:
                        pvals = None
                    else:
                        if self.type == 'Discrete':
                            pvals = [str(output[t]) for t in parents]
                        else:
                            pvals = [output[t] for t in parents]
                    output[node.name] = node.choose(self.distributions[node.name], pvals=pvals)
            seq.append(output)

        if as_df:
            return pd.DataFrame.from_dict(seq, orient='columns')
        else:
            return seq

    def predict(self, test: pd.DataFrame, parall_count: int = 1) -> Dict[str, Union[List[str], List[int], List[float]]]:
        """
        Function to predict columns from given data.
        Note that train data and test data must have different columns.
        Both train and test datasets must be cleaned from NaNs.

        Args:
            test (pd.DataFrame): test dataset
            parall_count (int, optional):number of threads. Defaults to 1.

        Returns:
            predicted data (dict): dict with column as key and predicted data as value
        """
        from joblib import Parallel, delayed

        def wrapper(bn: HybridBN, test: pd.DataFrame, columns: List[str]):
            preds = {column_name: list() for column_name in columns}

            if len(test) == 1:
                for i in range(test.shape[0]):
                    test_row = dict(test.iloc[i, :])
                    for n, key in enumerate(columns):
                        try:
                            sample = bn.sample(2000, evidence=test_row)
                            if bn[key].type.startswith(('Discrete', 'Logit', 'ConditionalLogit',)):
                                count_stats = sample[key].value_counts()
                                preds[key].append(count_stats.index[0])
                            else:
                                if bn.descriptor['signs'][key] == 'pos':
                                    sample = sample.loc[sample[key] >= 0]
                                if sample.shape[0] == 0:
                                    preds[key].append(np.nan)
                                else:
                                    pred = np.mean(sample[key].values)
                                    preds[key].append(pred)
                        except Exception as ex:
                            logger_network.error(ex)
                            preds[key].append(np.nan)
                return preds
            else:
                logger_network.error('Wrapper for one row from pandas.DataFrame')
                return {}

        columns = list(set(self.nodes_names) - set(test.columns.to_list()))
        if not columns:
            logger_network.error("Test data is the same as train.")
            return {}

        preds = {column_name: list() for column_name in columns}

        processed_list = Parallel(n_jobs=parall_count)(
            delayed(wrapper)(self, test.loc[[i]], columns) for i in tqdm(test.index, position=0, leave=True))

        for i in range(test.shape[0]):
            curr_pred = processed_list[i]
            for n, key in enumerate(columns):
                preds[key].append(curr_pred[key][0])

        for column in columns:
            preds[column] = [k for k in preds[column] if not pd.isna(k)]

        return preds

    def set_classifiers(self, classifiers: Dict[str, object]):
        """
        Set classifiers for logit nodes.
        classifiers: dict with node_name and Classifier
        """
        if not self.has_logit:
            logger_network.error("Logit nodes are forbidden.")
            return None

        for node in self.nodes:
            if "Logit" in node.type:
                if node.name in classifiers.keys():
                    node.classifier = classifiers[node.name]
                    node.type = re.sub(r"\([\s\S]*\)", f"({type(node.classifier).__name__})", node.type)
                else:
                    continue

    def plot(self, output: str):
        """
        Visualize a Bayesian Network. Result will be saved
        in parent directory in folder visualization_result.
        output: str name of output file
        """
        if not output.endswith('.html'):
            logger_network.error("This version allows only html format.")
            return None

        G = nx.DiGraph()
        nodes = [node.name for node in self.nodes]
        G.add_nodes_from(nodes)
        G.add_edges_from(self.edges)

        network = Network(height="800px", width="100%", notebook=True, directed=nx.is_directed(G),
                          layout='hierarchical')

        nodes_sorted = np.array(list(nx.topological_generations(G)), dtype=object)

        # Qualitative class of colormaps
        q_classes = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20',
                     'tab20b', 'tab20c']

        hex_colors = []
        for cls in q_classes:
            rgb_colors = plt.get_cmap(cls).colors
            hex_colors.extend(
                [matplotlib.colors.rgb2hex(rgb_color) for rgb_color in rgb_colors])

        hex_colors = np.array(hex_colors)

        # Number_of_colors in matplotlib in Qualitative class = 144

        class_number = len(
            set([node.type for node in self.nodes])
        )
        hex_colors_indexes = [random.randint(0, len(hex_colors) - 1) for _ in range(class_number)]
        hex_colors_picked = hex_colors[hex_colors_indexes]
        class2color = {cls: color for cls, color in zip(set([node.type for node in self.nodes]), hex_colors_picked)}
        name2class = {node.name: node.type for node in self.nodes}

        for level in range(len(nodes_sorted)):
            for node_i in range(len(nodes_sorted[level])):
                name = nodes_sorted[level][node_i]
                cls = name2class[name]
                color = class2color[cls]
                network.add_node(name, label=name, color=color, size=45, level=level, font={'size': 36},
                                 title=f'Узел байесовской сети {name} ({cls})')

        for edge in G.edges:
            network.add_edge(edge[0], edge[1])

        network.hrepulsion(node_distance=300, central_gravity=0.5)

        import os
        if not (os.path.exists('visualization_result')):
            os.mkdir("visualization_result")

        return network.show(f'visualization_result/' + output)


class DiscreteBN(BaseNetwork):
    """
    Bayesian Network with Discrete Types of Nodes
    """

    def __init__(self):
        super(DiscreteBN, self).__init__()
        self.type = 'Discrete'
        self.scoring_function = ""
        self._allowed_dtypes = ['disc', 'disc_num']
        self.has_logit = None
        self.use_mixture = None


class ContinuousBN(BaseNetwork):
    """
    Bayesian Network with Continuous Types of Nodes
    """

    def __init__(self, use_mixture: bool = False):
        super(ContinuousBN, self).__init__()
        self.type = 'Continuous'
        self._allowed_dtypes = ['cont']
        self.has_logit = None
        self.use_mixture = use_mixture
        self.scoring_function = ""


class HybridBN(BaseNetwork):
    """
    Bayesian Network with Mixed Types of Nodes
    """

    def __init__(self, has_logit: bool = False, use_mixture: bool = False):
        super(HybridBN, self).__init__()
        self._allowed_dtypes = ['cont', 'disc', 'disc_num']
        self.type = 'Hybrid'
        self.has_logit = has_logit
        self.use_mixture = use_mixture

    def validate(self, descriptor: Dict[str, Dict[str, str]]) -> bool:
        types = descriptor['types']
        s = set(types.values())
        return True if ({'cont', 'disc', 'disc_num'} == s) or ({'cont', 'disc'} == s) or ({'cont', 'disc_num'} == s) else False
