import time

start = time.time()

from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import bamt.Networks as Networks

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("../Data/real data/hack_processed_with_rf.csv")

cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
h = h[cols]

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder)])
discretized_data, est = p.apply(h)
info = p.info

bn = Networks.HybridBN()
bn.add_nodes(descriptor=info)

params = {'init_edges': [('Structural setting', 'Gross'), ('Netpay', 'Porosity'), ('Permeability', 'Depth')]}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',), params=params)
bn.get_info()
t1 = time.time()
bn.fit_parameters(data=h)
t2 = time.time()
print(f'PL elaspsed: {t2-t1}')

for num, el in enumerate(bn.sample(20), 1):
    print('\n', num)
    for name, val in el.items():
        print(f"{name: <15}", val)
