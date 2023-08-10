from bamt.networks.hybrid_bn import HybridBN
from bamt.preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp

data = pd.read_csv("../../data/real data/vk_data.csv").iloc[:1000, :10]
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
bn = HybridBN()

# add nodes to network
bn.add_nodes(info)

# using mutual information as scoring function for structure learning
bn.add_edges(discretized_data, scoring_function=('K2',))

bn.get_info(as_df=False)
bn.plot("entire.html")
plot_to = "family.html"

bn.find_family("has_high_education", height=1, depth=1, plot_to=plot_to)
