
import time

start = time.time()

from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import bamt.Networks as Networks

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("data/real data/hack_processed_with_rf.csv")
cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross', 'Netpay', 'Porosity', 'Permeability',
        'Depth']
h = h[cols]

# ROWS = 50
# h = h.iloc[:ROWS, :]

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

print("#" * 1000)
discretized_data, est = p.apply(h)
info = p.info

bn = Networks.ContinuousBN()

bn.add_nodes(descriptor=info)  # Error

# -----------
nodes_type_mixed = p.get_nodes_types(h)
columns = [col for col in h.columns.to_list() if not nodes_type_mixed[col] in ['disc', 'disc_num']]  # GET ONLY CONT
discrete_data = h[columns]

discretized_data, est = p.apply(discrete_data)  # info
info = p.info

bn = Networks.ContinuousBN()

bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',))

bn = Networks.ContinuousBN(use_mixture=True)

bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',))
bn.get_info()
