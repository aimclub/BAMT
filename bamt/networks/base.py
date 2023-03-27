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
from pyvis.network import Network
from pyitlib import discrete_random_variable as drv

from bamt.builders import ParamDict
from bamt.log import logger_network
from bamt.config import config

from bamt.nodes.base import BaseNode
from bamt.networks.hybrid_bn import HybridBN
import bamt.builders as Builders

from typing import Dict, Tuple, List, Callable, Optional, Type, Union, Any, Sequence

STORAGE = config.get('NODES', 'models_storage',
                     fallback='models_storage is not defined')


class BaseNetwork(object):
    """
    Base class for Bayesian Network
    """

    def __init__(self):
        """
        nodes: a list of nodes instances
        edges: a list of edges
        distributions: dict
        """
        self.type = 'Abstract'
        self._allowed_dtypes = ['Abstract']
        self.nodes = []
        self.edges = []
        self.weights = {}
        self.descriptor = {"types": {},
                           "signs": {}}
        self.distributions = {}
        self.has_logit = False
        self.use_mixture = False

    @property
    def nodes_names(self) -> List[str]:
        return [node.name for node in self.nodes]

    def __getitem__(self, node_name: str) -> Type[BaseNode]:
        if node_name in self.nodes_names:
            index = self.nodes_names.index(node_name)
            return self.nodes[index]

    def validate(self, descriptor: Dict[str, Dict[str, str]]) -> bool:
        types = descriptor['types']
        return True if all(
            [a in self._allowed_dtypes for a in types.values()]) else False

    def update_descriptor(self):
        new_nodes_names = [node.name for node in self.nodes]
        self.descriptor['types'] = {
            node: type for node,
            type in self.descriptor['types'].items() if node in new_nodes_names}
        if "cont" in self.descriptor["types"].values():
            self.descriptor['signs'] = {
                node: sign for node,
                sign in self.descriptor['signs'].items() if node in new_nodes_names}

    def add_nodes(self, descriptor: Dict[str, Dict[str, str]]):
        """
        Function for initializing nodes in Bayesian Network
        descriptor: dict with types and signs of nodes
        """
        if not self.validate(descriptor=descriptor):
            if not self.type == 'Hybrid':
                logger_network.error(
                    f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
                return
            else:
                logger_network.error(
                    f"Descriptor validation failed due to wrong type of column(s).")
                return
        elif ['Abstract'] in self._allowed_dtypes:
            return None
        self.descriptor = descriptor
        # LEVEL 1
        worker_1 = Builders.VerticesDefiner(descriptor, regressor=None)
        self.nodes = worker_1.vertices

    def add_edges(self,
                  data: pd.DataFrame,
                  scoring_function: Union[Tuple[str,
                  Callable],
                  Tuple[str]],
                  progress_bar: bool = True,
                  classifier: Optional[object] = None,
                  regressor: Optional[object] = None,
                  params: Optional[ParamDict] = None,
                  optimizer: str = 'HC'):
        """
        Base function for Structure learning
        scoring_function: tuple with the following format (NAME, scoring_function) or (NAME,)
        Params:
        init_edges: list of tuples, a graph to start learning with
        remove_init_edges: allows changes in a model defined by user
        white_list: list of allowed edges
        """
        if not self.has_logit and classifier:
            logger_network.error(
                "Classifiers dict will be ignored since logit nodes are forbidden.")
            return None

        # params validation
        if params:
            # init_edges validation
            if not self.has_logit and "init_edges" in params.keys():
                type_map = np.array([
                    [self.descriptor["types"][node1], self.descriptor["types"][node2]] for node1, node2 in
                    params["init_edges"]]
                )
                failed = (
                        (type_map[:, 0] == "cont") &
                        ((type_map[:, 1] == "disc") |
                         (type_map[:, 1] == "disc_num"))
                )
                if sum(failed):
                    logger_network.warning(
                        f"Edges between continuous nodes and disc nodes are forbidden (has_logit = {self.has_logit}), "
                        f"they will be ignored. Indexes: {np.where(failed)[0]}")
                    params["init_edges"] = [params["init_edges"][i] for i in range(
                        len(params["init_edges"])) if i not in np.where(failed)[0]]

        if not self.validate(descriptor=self.descriptor):
            logger_network.error(
                f"{self.type} BN does not support {'discrete' if self.type == 'Continuous' else 'continuous'} data")
            return None
        if optimizer == 'HC':
            worker = Builders.HCStructureBuilder(
                data=data,
                descriptor=self.descriptor,
                scoring_function=scoring_function,
                has_logit=self.has_logit,
                use_mixture=self.use_mixture,
                regressor=regressor)

            self.sf_name = scoring_function[0]

            worker.build(
                data=data,
                params=params,
                classifier=classifier,
                regressor=regressor,
                progress_bar=progress_bar)

            # update family
            self.nodes = worker.skeleton['V']
            self.edges = worker.skeleton['E']

    def calculate_weights(self, discretized_data: pd.DataFrame):
        """
        Provide calculation of link strength according mutual information between node and its parent(-s) values.
        """
        import bamt.utils.GraphUtils as gru
        if not all([i in ['disc', 'disc_num']
                    for i in gru.nodes_types(discretized_data).values()]):
            logger_network.error(
                f"calculate_weghts() method deals only with discrete data. Continuous data: " +
                f"{[col for col, type in gru.nodes_types(discretized_data).items() if type not in ['disc', 'disc_num']]}")
        if not self.edges:
            logger_network.error(
                "Bayesian Network hasn't fitted yet. Please add edges with add_edges() method")
        if not self.nodes:
            logger_network.error(
                "Bayesian Network hasn't fitted yet. Please add nodes with add_nodes() method")
        weights = dict()

        for node in self.nodes:
            parents = node.cont_parents + node.disc_parents
            if parents is None:
                continue
            y = discretized_data[node.name].values
            if len(parents) == 1:
                x = discretized_data[parents[0]].values
                ls_true = drv.information_mutual(X=y, Y=x)
                entropy = drv.entropy(X=y)
                weight = ls_true / entropy
                weights[(parents[0], node.name)] = weight
            else:
                for parent_node in parents:
                    x = discretized_data[parent_node].values
                    other_parents = [
                        tmp for tmp in parents if tmp != parent_node]
                    z = list()
                    for other_parent in other_parents:
                        z.append(list(discretized_data[other_parent].values))
                    ls_true = np.average(drv.information_mutual_conditional(
                        X=y, Y=x, Z=z, cartesian_product=True))
                    entropy = np.average(drv.entropy_conditional(
                        X=y, Y=z, cartesian_product=True)) + 1e-8
                    weight = ls_true / entropy
                    weights[(parent_node, node.name)] = weight
        self.weights = weights

    def set_nodes(self, nodes: List, info: Optional[Dict] = None):
        """
        additional function to set nodes manually. User should be aware that
        nodes must be a subclass of BaseNode.
        Params:
            nodes: dict with name and node (if a lot of nodes should be added)
            info: descriptor
        """
        if not info and not self.descriptor["types"]:
            logger_network.error(
                "In case of manual setting nodes user should set map for them as well.")
            return
        self.nodes = []
        for node in nodes:
            try:
                assert issubclass(type(node), BaseNode)
                self.nodes.append(node)
                continue
            except AssertionError:
                logger_network.error(
                    f"{node} is not an instance of {BaseNode}")
                continue
            except TypeError:
                logger_network.error(f"TypeError : {node.__class__}")
                continue
        if info:
            self.descriptor = info

    def set_edges(self, edges: Optional[List[Sequence[str]]] = None):
        """
        additional function to set edges manually. User should be aware that
        nodes must be a subclass of BaseNode.
        :param edges dict with name and node (if a lot of nodes should be added)
        """

        if not self.nodes:
            logger_network.error("Graph without nodes")
        self.edges = []
        for node1, node2 in edges:
            if isinstance(node1, str) and isinstance(node2, str):
                if self[node1] and self[node2]:
                    if not self.has_logit and \
                            self.descriptor["types"][node1] == "cont" and \
                            self.descriptor["types"][node2] == "disc":
                        logger_network.warning(
                            f"Restricted edge detected (has_logit=False) : [{node1}, {node2}]")
                        continue
                    else:
                        self.edges.append((node1, node2))
                else:
                    logger_network.error(f"Unknown nodes : [{node1}, {node2}]")
                    continue
            else:
                logger_network.error(
                    f"Unknown node(s) type: [{node1.__class__}, {node2.__class__}]")
                continue
        self.update_descriptor()

    def set_structure(self,
                      info: Optional[Dict] = None,
                      nodes: Optional[List] = None,
                      edges: Optional[List[Sequence[str]]] = None,
                      overwrite: bool = True):
        """
        Function to set structure manually
        info: Descriptor
        nodes, edges:
        overwrite: use 2nd stage of defining or not
        """
        if nodes and (
                info or (
                self.descriptor["types"] and self.descriptor["signs"])):
            self.set_nodes(nodes=nodes, info=info)
        if edges:
            self.set_edges(edges=edges)
            if overwrite:
                builder = Builders.VerticesDefiner(
                    descriptor=self.descriptor, regressor=None)  # init worker
                builder.skeleton['V'] = builder.vertices  # 1 stage
                builder.skeleton['E'] = self.edges
                builder.get_family()
                if self.edges:
                    builder.overwrite_vertex(
                        has_logit=self.has_logit,
                        use_mixture=self.use_mixture,
                        classifier=None,
                        regressor=None)
                    self.set_nodes(nodes=builder.skeleton['V'])
                else:
                    logger_network.error("Empty set of edges")

    def _param_validation(self, params: Dict[str, Any]) -> bool:
        if all(self[i] for i in params.keys()):
            for name, info in params.items():
                try:
                    self[name].choose(node_info=info, pvals=[])
                except Exception as ex:
                    logger_network.error("Validation failed", exc_info=ex)
                    return False
            return True
        else:
            logger_network.error(
                "Param validation failed due to unknown nodes.")
            return False

    def set_parameters(self, parameters: Dict):
        if not self.nodes:
            logger_network.error("Failed on search of BN's nodes.")

        self.distributions = parameters

        for node, data in self.distributions.items():
            if "hybcprob" in data.keys():
                if "mixture" in self[node].type.lower():
                    continue
                else:
                    if "gaussian" in self[node].type.lower():
                        model_type = "regressor"
                    else:
                        model_type = "classifier"

                    model = None
                    for v in data["hybcprob"].values():
                        if v[model_type]:
                            model = v[model_type]
                            break
                        else:
                            continue
                    if not model:
                        logger_network.warning(
                            f"Classifier/regressor for {node} hadn't been used.")

                    self[node].type = re.sub(
                        r"\([\s\S]*\)", f"({model})", self[node].type)

    def save_to_file(self, outdir: str, data: dict):
        """
        Function to save data to json file
        :param outdir: output directory
        :param data: dictionary to be saved
        """
        if not outdir.endswith('.json'):
            return None
        with open(outdir, 'w+') as out:
            json.dump(data, out)
        return True

    def save_params(self, outdir: str):
        """
        Function to save BN params to json file
        outdir: output directory
        """
        return self.save_to_file(outdir, self.distributions)

    def save_structure(self, outdir: str):
        """
        Function to save BN edges to json file
        outdir: output directory
        """
        return self.save_to_file(outdir, self.edges)

    def save(self, outdir: str):
        """
        Function to save the whole BN to json file
        :param outdir: output directory
        """
        new_weights = {str(key): self.weights[key] for key in self.weights}
        outdict = {
            'info': self.descriptor,
            'edges': self.edges,
            'parameters': self.distributions,
            'weights': new_weights
        }
        return self.save_to_file(outdir, outdict)

    def load(self, input_dir: str):
        """
        Function to load the whole BN from json file
        :param input_dir: input directory
        """

        class CompatibilityError(Exception):
            def __init__(self, type: str):
                """
                Args:
                    type: either use_mixture or has_logit is wrong
                """
                super().__init__(
                    f"This parameter is not the same as father's parameter: {type}")

        with open(input_dir) as f:
            input_dict = json.load(f)

        self.add_nodes(input_dict['info'])
        self.set_structure(edges=input_dict['edges'])

        # check compatibility with father network.
        if not self.use_mixture:
            for node_data in input_dict['parameters'].values():
                if 'hybcprob' not in node_data.keys():
                    continue
                else:
                    # Since we don't have information about types of nodes, we
                    # should derive it from parameters.
                    if any(list(node_keys.keys()) == ["covars", "mean", "coef"]
                           for node_keys in node_data['hybcprob'].values()):
                        raise CompatibilityError("use_mixture")

        if not self.has_logit:
            if not all(
                    ob1 == [
                        ob2[0],
                        ob2[1]] for ob1,
                    ob2 in zip(
                        input_dict['edges'],
                        self.edges)):
                raise CompatibilityError("has_logit")

        self.set_parameters(parameters=input_dict['parameters'])
        str_keys = list(input_dict['weights'].keys())
        tuple_keys = [eval(key) for key in str_keys]
        weights = {}
        for tuple_key in tuple_keys:
            weights[tuple_key] = input_dict['weights'][str(tuple_key)]
        self.weights = weights

    def fit_parameters(self, data: pd.DataFrame, dropna: bool = True):
        """
        Base function for parameter learning
        """
        if dropna:
            data = data.dropna()
            data.reset_index(inplace=True, drop=True)

        if not os.path.isdir(STORAGE):
            os.makedirs(STORAGE)

        # init folder
        if not os.listdir(STORAGE):
            os.makedirs(os.path.join(STORAGE, "0"))

        index = sorted(
            [int(id) for id in os.listdir(STORAGE)]
        )[-1] + 1
        os.makedirs(os.path.join(STORAGE, str(index)))

        # Turn all discrete values to str for learning algorithm
        if 'disc_num' in self.descriptor['types'].values():
            columns_names = [name for name, t in self.descriptor['types'].items() if t in [
                'disc_num']]
            data[columns_names] = data.loc[:, columns_names].astype('str')

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
                parents_types.append([self.descriptor['types'][name]
                                      for name in n.cont_parents + n.disc_parents])
                parents.append(
                    [name for name in n.cont_parents + n.disc_parents])
            return pd.DataFrame({'name': names, 'node_type': types_n,
                                 'data_type': types_d, 'parents': parents,
                                 'parents_types': parents_types})
        else:
            for n in self.nodes:
                print(
                    f"{n.name: <20} | {n.type: <50} | {self.descriptor['types'][n.name]: <10} | {str([self.descriptor['types'][name] for name in n.cont_parents + n.disc_parents]): <50} | {str([name for name in n.cont_parents + n.disc_parents])}")

    def sample(self,
               n: int,
               models_dir: Optional[str] = None,
               progress_bar: bool = True,
               evidence: Optional[Dict[str, Union[str, int, float]]] = None,
               as_df: bool = True,
               predict: bool = False,
               parall_count: int = 1) -> \
            Union[None, pd.DataFrame, List[Dict[str, Union[str, int, float]]]]:
        """
        Sampling from Bayesian Network
        n: int number of samples
        evidence: values for nodes from user
        parall_count: number of threads. Defaults to 1.
        """
        from joblib import Parallel, delayed

        random.seed()
        if not self.distributions.items():
            logger_network.error(
                "Parameter learning wasn't done. Call fit_parameters method")
            return None
        if evidence:
            for node in self.nodes:
                if (node.type == 'Discrete') & (node.name in evidence.keys()):
                    if not (isinstance(evidence[node.name], str)):
                        evidence[node.name] = str(int(evidence[node.name]))

        def wrapper():
            output = {}
            for node in self.nodes:
                parents = node.cont_parents + node.disc_parents
                if evidence and node.name in evidence.keys():
                    output[node.name] = evidence[node.name]
                else:
                    if not parents:
                        pvals = None
                    else:
                        if self.type == 'Discrete':
                            pvals = [str(output[t]) for t in parents]
                        else:
                            pvals = [output[t] for t in parents]

                        # If any nan from parents, sampling from node blocked.
                        if any(pd.isnull(pvalue) for pvalue in pvals):
                            output[node.name] = np.nan
                            continue

                    node_data = self.distributions[node.name]
                    if models_dir and ("hybcprob" in node_data.keys()):
                        for obj, obj_data in node_data["hybcprob"].items():
                            if "serialization" in obj_data.keys():
                                if "gaussian" in node.type.lower():
                                    model_type = "regressor"
                                else:
                                    model_type = "classifier"
                                if obj_data["serialization"] == 'joblib' and obj_data[
                                    f"{model_type}_obj"]:
                                    new_path = models_dir + \
                                               f"\\{node.name.replace(' ', '_')}\\{obj}.joblib.compressed"
                                    node_data["hybcprob"][obj][f"{model_type}_obj"] = new_path

                    if predict:
                        output[node.name] = \
                            node.predict(node_data, pvals=pvals)
                    else:
                        output[node.name] = \
                            node.choose(node_data, pvals=pvals)
            return output

        if progress_bar:
            seq = Parallel(
                n_jobs=parall_count)(
                delayed(wrapper)() for _ in tqdm(
                    range(n),
                    position=0,
                    leave=True))
        else:
            seq = Parallel(
                n_jobs=parall_count)(
                delayed(wrapper)() for _ in range(n))
        seq_df = pd.DataFrame.from_dict(seq, orient='columns')
        seq_df.dropna(inplace=True)
        cont_nodes = [c.name for c in self.nodes if c.type !=
                      'Discrete' and 'Logit' not in c.type]
        positive_columns = [
            c for c in cont_nodes if self.descriptor['signs'][c] == 'pos']
        seq_df = seq_df[(seq_df[positive_columns] >= 0).all(axis=1)]
        seq_df.reset_index(inplace=True, drop=True)
        seq = seq_df.to_dict('records')

        if as_df:
            return pd.DataFrame.from_dict(seq, orient='columns')
        else:
            return seq

    def predict(self,
                test: pd.DataFrame,
                parall_count: int = 1,
                progress_bar: bool = True) -> Dict[str,
    Union[List[str],
    List[int],
    List[float]]]:
        """
        Function to predict columns from given data.
        Note that train data and test data must have different columns.
        Both train and test datasets must be cleaned from NaNs.

        Args:
            test (pd.DataFrame): test dataset
            parall_count (int, optional):number of threads. Defaults to 1.
            progress_bar: verbose mode.

        Returns:
            predicted data (dict): dict with column as key and predicted data as value
        """
        if test.isnull().any().any():
            logger_network.error("Test data contains NaN values.")
            return {}

        from joblib import Parallel, delayed

        def wrapper(bn: HybridBN, test: pd.DataFrame, columns: List[str]):
            preds = {column_name: list() for column_name in columns}

            if len(test) == 1:
                for i in range(test.shape[0]):
                    test_row = dict(test.iloc[i, :])
                    for n, key in enumerate(columns):
                        try:
                            sample = bn.sample(
                                1, evidence=test_row, predict=True, progress_bar=False)
                            if sample.empty:
                                preds[key].append(np.nan)
                                continue
                            if bn.descriptor['types'][key] == 'cont':
                                if (bn.descriptor['signs'][key] == 'pos') & (
                                        sample.loc[0, key] < 0):
                                    # preds[key].append(np.nan)
                                    preds[key].append(0)
                                else:
                                    preds[key].append(sample.loc[0, key])
                            else:
                                preds[key].append(sample.loc[0, key])
                        except Exception as ex:
                            logger_network.error(ex)
                            preds[key].append(np.nan)
                return preds
            else:
                logger_network.error(
                    'Wrapper for one row from pandas.DataFrame')
                return {}

        columns = list(set(self.nodes_names) - set(test.columns.to_list()))
        if not columns:
            logger_network.error("Test data is the same as train.")
            return {}

        preds = {column_name: list() for column_name in columns}

        if progress_bar:
            processed_list = Parallel(n_jobs=parall_count)(delayed(wrapper)(
                self, test.loc[[i]], columns) for i in tqdm(test.index, position=0, leave=True))
        else:
            processed_list = Parallel(n_jobs=parall_count)(
                delayed(wrapper)(self, test.loc[[i]], columns) for i in test.index)

        for i in range(test.shape[0]):
            curr_pred = processed_list[i]
            for n, key in enumerate(columns):
                preds[key].append(curr_pred[key][0])

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
                    node.type = re.sub(
                        r"\([\s\S]*\)",
                        f"({type(node.classifier).__name__})",
                        node.type)
                else:
                    continue

    def set_regressor(self, regressors: Dict[str, object]):
        """
        Set classifiers for gaussian nodes.
        classifiers: dict with node_name and Classifier
        """

        for node in self.nodes:
            if "Gaussian" in node.type:
                if node.name in regressors.keys():
                    node.regressor = regressors[node.name]
                    node.type = re.sub(
                        r"\([\s\S]*\)",
                        f"({type(node.regressor).__name__})",
                        node.type)
                else:
                    continue

    def plot(self, output: str):
        """
        Visualize a Bayesian Network. Result will be saved
        in the parent directory in folder visualization_result.
        output: str name of output file
        """
        if not output.endswith('.html'):
            logger_network.error("This version allows only html format.")
            return None

        G = nx.DiGraph()
        nodes = [node.name for node in self.nodes]
        G.add_nodes_from(nodes)
        G.add_edges_from(self.edges)

        network = Network(
            height="800px",
            width="100%",
            notebook=True,
            directed=nx.is_directed(G),
            layout='hierarchical')

        nodes_sorted = np.array(
            list(nx.topological_generations(G)), dtype=object)

        # Qualitative class of colormaps
        q_classes = [
            'Pastel1',
            'Pastel2',
            'Paired',
            'Accent',
            'Dark2',
            'Set1',
            'Set2',
            'Set3',
            'tab10',
            'tab20',
            'tab20b',
            'tab20c']

        hex_colors = []
        for cls in q_classes:
            rgb_colors = plt.get_cmap(cls).colors
            hex_colors.extend([matplotlib.colors.rgb2hex(rgb_color)
                               for rgb_color in rgb_colors])

        hex_colors = np.array(hex_colors)

        # Number_of_colors in matplotlib in Qualitative class = 144

        class_number = len(
            set([node.type for node in self.nodes])
        )
        hex_colors_indexes = [random.randint(
            0, len(hex_colors) - 1) for _ in range(class_number)]
        hex_colors_picked = hex_colors[hex_colors_indexes]
        class2color = {cls: color for cls, color in zip(
            set([node.type for node in self.nodes]), hex_colors_picked)}
        name2class = {node.name: node.type for node in self.nodes}

        for level in range(len(nodes_sorted)):
            for node_i in range(len(nodes_sorted[level])):
                name = nodes_sorted[level][node_i]
                cls = name2class[name]
                color = class2color[cls]
                network.add_node(name, label=name, color=color, size=45, level=level, font={
                    'size': 36}, title=f'Узел байесовской сети {name} ({cls})')

        for edge in G.edges:
            network.add_edge(edge[0], edge[1])

        network.hrepulsion(node_distance=300, central_gravity=0.5)

        if not (os.path.exists('visualization_result')):
            os.mkdir("visualization_result")

        return network.show(f'visualization_result/' + output)
