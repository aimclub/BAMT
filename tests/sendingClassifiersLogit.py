import json

import bamt.Networks as Nets
import bamt.Preprocessors as preprocessors

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pp

import pandas as pd

hack_data = pd.read_csv("../data/real data/hack_processed_with_rf.csv")[
    ['Tectonic regime', 'Period', 'Lithology', 'Structural setting',
     'Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']]

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

p = preprocessors.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

discretized_data, est = p.apply(hack_data)

bn = Nets.HybridBN(has_logit=True)
info = p.info

with open(r"C:\Users\Roman\Desktop\mymodels\mynet.json") as f:
    net_data = json.load(f)

bn.add_nodes(net_data["info"])
bn.set_structure(edges=net_data["edges"])
bn.set_parameters(net_data["parameters"])

print(bn.sample(10, models_dir=r"<new dir>"))

# bn.add_nodes(info)
#
# bn.add_edges(discretized_data, scoring_function=("BIC",))
# bn.set_classifiers(classifiers={'Structural setting': DecisionTreeClassifier(),
#                                 'Lithology': RandomForestClassifier(),
#                                 'Period': KNeighborsClassifier(n_neighbors=2)})
#
# bn.fit_parameters(hack_data)
#
# bn.save("mynet.json")

bn.get_info(as_df=False)