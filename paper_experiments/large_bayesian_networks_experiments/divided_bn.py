import sys, os, ast
import prince
import pandas as pd
import numpy as np
import bamt.preprocessors as pp
import scipy.stats as stats
from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.discrete_bn import DiscreteBN
from bamt.networks.hybrid_bn import HybridBN
from collections import defaultdict
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from varclushi import VarClusHi
from pgmpy.estimators import K2Score
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features
    :param data: data for encoding
    :return: encoded data
    """
    for col in data.columns:
        if data[col].dtype == 'object':
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

def varclushi_clustering(
        data: pd.DataFrame,
        maxeigval2: int = 1,
        maxclus: int = 4) -> dict:
    data = encode_categorical_features(data)
    vc = VarClusHi(data, maxeigval2=maxeigval2, maxclus=maxclus)
    vc.varclus()
    clusters_df = vc.rsquare
    clusters_df = clusters_df[['Cluster', 'Variable']]
    clusters = {}
    for i in range(max(clusters_df['Cluster']) + 1):
        clusters[i] = list(
            clusters_df[clusters_df['Cluster'] == i]['Variable'])
    return clusters

def kmeans_clustering(data, n_clusters, max_cluster_size=50, scaling=True):
    # Prepare the data
    if scaling:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.T)
    else:
        data_scaled = data.T

    # Perform KMeans clustering on scaled data
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_scaled)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Create a dictionary with clusters
    clustered_variables = defaultdict(list)
    for j in range(len(data.columns)):
        clustered_variables[cluster_labels[j]].append(data.columns[j])

    cluster_centers = kmeans.cluster_centers_

    distribute_extra_variables(
        clustered_variables,
        cluster_centers,
        data_scaled,
        max_cluster_size,
        data)

    return clustered_variables

def distribute_extra_variables(
        clustered_variables,
        cluster_centers,
        coordinates,
        max_cluster_size,
        data):
    overflow = []
    for key, value in clustered_variables.items():
        if len(value) > max_cluster_size:
            overflow.extend((key, data.columns.get_loc(index))
                            for index in value[max_cluster_size:])
            clustered_variables[key] = value[:max_cluster_size]

    if overflow:
        for original_cluster, index in overflow:
            var_coordinate = coordinates.iloc[index]
            distances = [
                np.linalg.norm(
                    var_coordinate -
                    center) for center in cluster_centers]
            # Prevent returning the variable to the same cluster
            distances[original_cluster] = float('inf')
            closest_cluster = np.argmin(distances)

            # Find a cluster that has space and is closest to the variable
            while len(
                    clustered_variables[closest_cluster]) >= max_cluster_size:
                distances[closest_cluster] = float('inf')
                closest_cluster = np.argmin(distances)

            clustered_variables[closest_cluster].append(data.columns[index])

def mca_clustering(data, n_clusters, max_cluster_size=50):
    # Perform MCA
    mca = prince.MCA()
    mca.fit(data)

    # Get transformed coordinates
    coordinates = mca.column_coordinates(data)

    # Perform KMeans clustering on coordinates
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coordinates)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Create a dictionary with clusters
    clustered_variables = defaultdict(list)
    for j in range(len(data.columns)):
        clustered_variables[cluster_labels[j]].append(data.columns[j])

    cluster_centers = kmeans.cluster_centers_

    distribute_extra_variables(
        clustered_variables,
        cluster_centers,
        coordinates,
        max_cluster_size,
        data)

    return clustered_variables


def famd_clustering(data, n_clusters, max_cluster_size=50):
    # Perform FAMD
    famd = prince.FAMD()
    famd.fit(data)

    # Get transformed coordinates
    coordinates = famd.column_coordinates(data)

    # Perform KMeans clustering on coordinates
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coordinates)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Create a dictionary with clusters
    clustered_variables = defaultdict(list)
    for j in range(len(data.columns)):
        clustered_variables[cluster_labels[j]].append(data.columns[j])

    cluster_centers = kmeans.cluster_centers_

    distribute_extra_variables(
        clustered_variables,
        cluster_centers,
        coordinates,
        max_cluster_size,
        data)

    return clustered_variables


class DividedBN:

    def __init__(self,
                 data: pd.DataFrame,
                 data_type: str = 'mixed',
                 max_local_structures: int = 8,
                 hidden_nodes_clusters=None):
        """
        :param data: data for clustering
        :param cluster_number: number of clusters
        :param max_var_number_in_cluster: maximum number of variables in cluster
        """
        self.data = data
        self.data_type = data_type
        self.max_local_structures = max_local_structures
        self.local_structures_nodes = {}
        self.local_structures_edges = {}
        self.hidden_nodes_clusters = hidden_nodes_clusters
        self.hidden_nodes = {}
        self.local_structures_info = {}
        self.root_nodes = {}
        self.child_nodes = {}
        self.external_edges = {}

    def set_local_structures(self,
                             has_logit: bool = True,
                             use_mixture: bool = True,
                             maxeigval2: int = 1,
                             parallel_count: int = -1):

        def create_bn(has_logit, use_mixture):
            if self.data_type == "mixed":
                return HybridBN(
                    has_logit=has_logit,
                    use_mixture=use_mixture)
            elif self.data_type == "discrete":
                return DiscreteBN()
            elif self.data_type == "continuous":
                return ContinuousBN()

        def find_root_and_child_nodes(local_structure_info):
            root_nodes = local_structure_info[local_structure_info['parents'].str.len(
            ) == 0]['name'].tolist()
            list_of_all_parents = sum(
                local_structure_info['parents'].tolist(), [])
            child_nodes = [
                node for node in local_structure_info['name'] if node not in list_of_all_parents]
            return root_nodes, child_nodes

        def process_key(key):
            data_cluster = self.data[self.local_structures_nodes[key]]
            discretized_merged_data, info = self.preprocess_data(data_cluster)
            bn = create_bn(has_logit, use_mixture)
            bn.add_nodes(info)
            bn.add_edges(
                discretized_merged_data,
                scoring_function=(
                    'K2',
                    K2Score))
            local_structure_info = bn.get_info()
            root_nodes, child_nodes = find_root_and_child_nodes(
                local_structure_info)
            return key, bn.edges, local_structure_info, root_nodes, child_nodes

        if self.data_type == "continuous":
            self.local_structures_nodes = varclushi_clustering(
                self.data, maxeigval2=maxeigval2, maxclus=self.max_local_structures)
        elif self.data_type == "discrete":
            self.local_structures_nodes = mca_clustering(
                self.data, self.max_local_structures)
        elif self.data_type == "mixed":
            self.local_structures_nodes = famd_clustering(
                self.data, self.max_local_structures)

        results = Parallel(n_jobs=parallel_count)(
            delayed(process_key)(key) for key in self.local_structures_nodes)

        for key, edges, local_structure_info, root_nodes, child_nodes in results:
            self.local_structures_edges[key] = edges
            self.local_structures_info[key] = local_structure_info
            self.root_nodes[key] = root_nodes
            self.child_nodes[key] = child_nodes

    def preprocess_data(self, data_cluster):
        encoder = preprocessing.LabelEncoder()
        discretizer = preprocessing.KBinsDiscretizer(
            n_bins=5, encode='ordinal', strategy='quantile')
        p = pp.Preprocessor(
            [('encoder', encoder), ('discretizer', discretizer)])
        discretized_merged_data, _ = p.apply(data_cluster)
        return discretized_merged_data, p.info

    def find_optimal_clusters(
            self,
            data,
            data_type,
            max_clusters=10,
            random_state=42):
        if data_type == "mixed":
            cat_columns = self.detect_categorical_columns(data)
        costs = []
        for n_clusters in range(1, max_clusters + 1):
            if data_type == "continuous":
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state)
            elif data_type == "discrete":
                model = KModes(
                    n_clusters=n_clusters,
                    init='Huang',
                    verbose=0,
                    n_jobs=-1)
            elif data_type == "mixed":
                model = KPrototypes(
                    n_clusters=n_clusters,
                    init='Cao',
                    verbose=0,
                    n_jobs=-1)

            model.fit(data)
            costs.append(model.cost_ if data_type !=
                         "continuous" else model.inertia_)

        # Find the elbow point
        elbow_point = np.argmax(np.diff(np.diff(costs))) + 2
        return elbow_point

    def set_hidden_nodes(self, data):
        for key in self.local_structures_nodes:
            data_cluster = data[self.local_structures_nodes[key]]

            if self.hidden_nodes_clusters is None:
                # Use the Elbow method to find the optimal number of clusters
                n_clusters = self.find_optimal_clusters(
                    data_cluster, self.data_type)
                print(
                    "Optimal number of clusters for local structure {} is {}".format(
                        key, n_clusters))
            else:
                n_clusters = self.hidden_nodes_clusters

            if self.data_type == "continuous":
                model = KMeans(n_clusters=n_clusters)
            elif self.data_type == "discrete":
                model = KMeans(n_clusters=n_clusters)
                data_cluster = encode_categorical_features(data_cluster)
                # model = KModes(
                #     n_clusters=n_clusters,
                #     init='Huang',
                #     verbose=0,
                #     n_jobs=-1)
            elif self.data_type == "mixed":
                cat_columns = self.detect_categorical_columns(data_cluster)
                model = KPrototypes(
                    n_clusters=n_clusters,
                    init='Cao',
                    verbose=0,
                    n_jobs=-1)

            model.fit(data_cluster)
            self.hidden_nodes[key] = model.fit_predict(
                data_cluster).astype(np.int32)

    def detect_categorical_columns(self, data):
        """
        Automatically detects categorical columns based on their data types.
        """
        categorical_columns = []
        for i, col in enumerate(data.columns):
            if not np.issubdtype(
                    data[col].dtype,
                    np.number) or np.issubdtype(
                    data[col].dtype,
                    np.integer):
                categorical_columns.append(i)
        return categorical_columns

    def connect_structures_hc(self, evolutionary_edges):
        def process_meta_edge(meta_edge, self):
            source_structure_key = int(str(meta_edge[0]))
            target_structure_key = int(str(meta_edge[1]))

            source_nodes = self.local_structures_nodes[source_structure_key]
            target_nodes = self.local_structures_nodes[target_structure_key]

            source_edges = self.local_structures_edges[source_structure_key]
            target_edges = self.local_structures_edges[target_structure_key]

            united_edges = source_edges + target_edges
            init_edges = [tuple(edge) for edge in united_edges]

            source_data = self.data[source_nodes]
            target_data = self.data[target_nodes]

            merged_data = pd.concat([source_data, target_data], axis=1)

            if self.data_type == "mixed":
                bn = HybridBN()
            elif self.data_type == "discrete":
                bn = DiscreteBN()
            elif self.data_type == "continuous":
                bn = ContinuousBN()

            discretized_merged_data, merged_info = self.preprocess_data(merged_data)

            bn.add_nodes(merged_info)

            all_possible_edges_between_s_t = [(i, j) for i in source_nodes for j in target_nodes]
            all_possible_edges_between_t_s = [(i, j) for i in target_nodes for j in source_nodes]

            white_list = all_possible_edges_between_s_t + all_possible_edges_between_t_s

            params = {
                'white_list': white_list,
                'init_edges': init_edges,
                'remove_init_edges': False
            }

            bn.add_edges(discretized_merged_data, scoring_function=('K2', K2Score), params=params)

            learned_edges = bn.edges

            learned_edges_list = [list(edge) for edge in learned_edges]

            return (meta_edge, learned_edges_list)

        external_edges = {}

        results = Parallel(n_jobs=-1)(delayed(process_meta_edge)(meta_edge, self) for meta_edge in evolutionary_edges)

        for meta_edge, learned_edges_list in results:
            external_edges[meta_edge] = learned_edges_list

        self.external_edges = external_edges
        return external_edges

    def connect_structures_simple(self, evolutionary_edges):
        external_edges = []
        for meta_edge in evolutionary_edges:
            source_structure = int(str(meta_edge[0]))
            target_structure = int(str(meta_edge[1]))

            for source_node in self.child_nodes[source_structure]:
                for target_node in self.root_nodes[target_structure]:
                    external_edges.append([source_node, target_node])

        return external_edges

    def connect_structures_spearman(self, evolutionary_edges, percentile_threshold=95, hard_threshold=0.9):
        correlations = []
        node_pairs = []
        meta_edges = []

        if self.data_type == "discrete" or self.data_type == "mixed":
            encoded_data = encode_categorical_features(self.data)
        else:
            encoded_data = self.data

        for source_structure, target_structure in evolutionary_edges:
            for source_node in self.local_structures_nodes[int(str(source_structure))]:
                for target_node in self.local_structures_nodes[int(str(target_structure))]:
                    source_data = encoded_data[str(source_node)]
                    target_data = encoded_data[str(target_node)]

                    correlation, _ = stats.spearmanr(source_data, target_data)
                    # print(f"Correlation between {source_node} and {target_node}: {correlation}")

                    correlations.append(correlation)
                    node_pairs.append((source_node, target_node))
                    meta_edges.append((source_structure, target_structure))

        if correlations:
            threshold = np.percentile(correlations, percentile_threshold)
            if threshold < hard_threshold:
                threshold = hard_threshold
            print(f"Threshold: {threshold}")

            external_edges = {}
            for i, (source_node, target_node) in enumerate(node_pairs):
                if correlations[i] >= threshold:
                    metaedge = meta_edges[i]
                    if metaedge not in external_edges:
                        external_edges[metaedge] = []
                    external_edges[metaedge].append((source_node, target_node))
                    print(f"Connected {source_node} and {target_node} with correlation {correlations[i]}")
        else:
            print("No valid correlations found")
            external_edges = {}

        return external_edges
