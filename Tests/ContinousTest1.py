import time

start = time.time()

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("../Data/hack_processed_with_rf.csv")
cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
h = h[cols]

# ROWS = 50
# h = h.iloc[:ROWS, :]

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

print("#"*1000)
discretized_data, est = p.apply(h)
info = p.info

bn = Networks.ContinuousBN()

bn.add_nodes(descriptor=info) # Error

#-----------
nodes_type_mixed = p.get_nodes_types(h)
columns = [col for col in h.columns.to_list() if not nodes_type_mixed[col] in ['disc','disc_num']] # GET ONLY CONT
discrete_data = h[columns]

discretized_data, est = p.apply(discrete_data) # warning
info = p.info
print(info)

bn = Networks.ContinuousBN()

bn.add_nodes(descriptor=info)

for node in bn.nodes:
    print(node.name, node.type)

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',), params=params)

print("-"*50)

bn = Networks.ContinuousBN(use_mixture=True)

bn.add_nodes(descriptor=info)

for node in bn.nodes:
    print(node.name, node.type)

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',), params=params)
# t1 = time.time()
# bn.fit_parameters(data=h)
# t2 = time.time()
# print(f'PL elaspsed: {t2-t1}')
# for node, d in bn.distributions.items():
#     print(node,":", d)
#     break