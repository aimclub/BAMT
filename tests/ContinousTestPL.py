# FIX IT
import sys
import os

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, path)
# ---------

import time

start = time.time()

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("../Data/real data/hack_processed_with_rf.csv")
cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross', 'Netpay', 'Porosity', 'Permeability',
        'Depth']
h = h[cols]

print(h.describe())

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

p = Preprocessor([('discretizer', discretizer)])

# -----------
nodes_type_mixed = p.get_nodes_types(h)
columns = [col for col in h.columns.to_list() if not nodes_type_mixed[col] in ['disc', 'disc_num']]  # GET ONLY CONT
discrete_data = h[columns]

discretized_data, est = p.apply(discrete_data)  # info
info = p.info

bn = Networks.ContinuousBN(use_mixture=True)  # use_mixture = False as well

bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',))
bn.get_info(as_df=False)
t1 = time.time()
bn.fit_parameters(data=h)
t2 = time.time()
print(f'PL elaspsed: {t2 - t1}')
# Without async: 0.00699925422668457
# With: 0.0019998550415039062
print('Improvement: %.d' % (0.00699925422668457 // 0.0019998550415039062))
# After rebuilding: 0.0


for num, el in enumerate(bn.sample(10, as_df=False), 1):
    print('\n', num)
    for name, val in el.items():
        print(f"{name: <15}", val)
