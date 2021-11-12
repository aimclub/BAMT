import time

start = time.time()

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks
from pgmpy.estimators import K2Score
from Utils import GraphUtils as gru

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

discretized_data, est = p.apply(vk_data)
info = p.info

bn = Networks.DiscreteBN()
bn.add_nodes(descriptor=info) # error

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('K2', K2Score), params=params) # error
#
# # --------------------

nodes_type_mixed = gru.nodes_types(vk_data)
columns = [col for col in vk_data.columns.to_list() if nodes_type_mixed[col] in ['disc','disc_num']] # GET ONLY DISCRETE
discrete_data = vk_data[columns]

discretized_data, est = p.apply(discrete_data) # warning
info = p.info

bn = Networks.DiscreteBN()
bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('K2', K2Score), params=params)
t1 = time.time()
bn.fit_parameters(data=vk_data)
t2 = time.time()
print(f'PL elaspsed: {t2-t1}')
for node, d in bn.distributions.items():
    print(node,":", d)
    break