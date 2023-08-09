from bamt.display.display import GraphAnalyzer

from bamt.networks.hybrid_bn import HybridBN
from bamt.preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp

data = pd.read_csv("../../data/real data/vk_data.csv").iloc[:1000, :20]
print(data.shape)
# print(data.columns)
# set encoder and discretizer
encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# create preprocessor object with encoder and discretizer
p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

# discretize data for structure learning
discretized_data, est = p.apply(data)

# get information about data
info = p.info

# initialize network object
bn = HybridBN(has_logit=True)

# add nodes to network
bn.add_nodes(info)

# using mutual information as scoring function for structure learning
bn.add_edges(discretized_data, scoring_function=('K2',))

bn.get_info(as_df=False)
# bn.plot("entire.html")
print(GraphAnalyzer(bn).markov_blanket(node_name="instagram", plot_to="family.html"))
