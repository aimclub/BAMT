from pyBN.learning.structure.score.random_restarts import hc_rr
from pomegranate import BayesianNetwork, DiscreteDistribution
from pyBN.classes.bayesnet import BayesNet
from sklearn.cluster import KMeans
import numpy as np

"""
Function for trainig the structure
and parameters of BN from data


Input:
-data
Source dataset with discretized variables

-cluster
Number of clusters for KMeans to fill
the hidden var

-init_nodes
list of initial nodes which
can't have parents


Output:
BayesianNetwork object


"""

def train_model(data: np.ndarray, clusters: int = 5, init_nodes: list = None) -> BayesianNetwork:
    
    bn = BayesNet()
    #Сluster the initial data in order to fill in a hidden variable based on the distribution of clusters
    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(data)
    labels = kmeans.labels_
    hidden_dist = DiscreteDistribution.from_samples(labels)
    hidden_var = np.array(hidden_dist.sample(data.shape[0]))
    new_data = np.column_stack((data, hidden_var))
    latent = (new_data.shape[1])-1

    #Train the network structure on data taking into account a hidden variable
    bn = hc_rr(new_data, latent = latent, init_nodes = init_nodes)
    structure = []
    nodes = sorted(list(bn.nodes()))
    for rv in nodes:
        structure.append(tuple(bn.F[rv]['parents']))
    structure = tuple(structure)
    bn = BayesianNetwork.from_structure(new_data, structure)
    bn.bake()
    #Learn a hidden variable
    hidden_var = np.array([np.nan] * (data.shape[0]))
    new_data = np.column_stack((data, hidden_var))
    bn.predict(new_data)
    bn.fit(new_data)
    bn.bake()
    return (bn)