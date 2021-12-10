# FIX IT
import sys
import os
path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, path)
#---------

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

print(h.describe())
print("-----")
p2 = time.time()
print(f"Time elapsed for preparing data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

#-----------
coded_data, est = p.apply(h)
info = p.info

bn = Networks.HybridBN(use_mixture=True) # use_mixture = <OPPOSITE> as well
bn.add_nodes(descriptor=info)

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}
bn.add_edges(data=coded_data, optimizer='HC', scoring_function=('MI',), params=params)

t1 = time.time()
bn.fit_parameters(data=h)
t2 = time.time()
print(f'PL elaspsed: {t2-t1}')

for num, el in enumerate(bn.sample(10), 1):
    print('\n',num)
    for name, val in el.items():
        print(f"{name: <15}", val)

# for num, el in enumerate(bn.sample(100), 1):
#     print(f"{num: <5}", [el[key] for key in list(bn.distributions.keys())[0:20]])