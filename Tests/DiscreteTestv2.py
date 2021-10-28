import time

start = time.time()

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks
from pgmpy.estimators import K2Score

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

vk_data = pd.read_csv("../Data/vk_data.csv")
ROWS = 50
vk_data = vk_data.iloc[:ROWS, :]

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

nodes_type_mixed = p.get_nodes_types(vk_data)

columns = [col for col in vk_data.columns.to_list() if nodes_type_mixed[col] in ['disc','disc_num']]
discrete_data = vk_data[columns]

discretized_data, est = p.apply(discrete_data)
info = p.get_info()

info = {"types": p.get_nodes_types(discretized_data),
        "signs": info['signs']}


bn = Networks.DiscreteBN()
bn.add_nodes(descriptor=info)

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('K2', K2Score), params=params)
bn.fit_parameters(data=discretized_data)
for node, d in bn.distributions.items():
    print(node,":", d)