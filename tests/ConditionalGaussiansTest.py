import sys
import os

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, path)
# ---------
import time

start = time.time()

from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from bamt import Networks

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("../data/hack_processed_with_rf.csv")
cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross', 'Netpay', 'Porosity', 'Permeability',
        'Depth']
h = h[cols]

print(h.describe())
print("-----")
p2 = time.time()
print(f"Time elapsed for preparing data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

# -----------
discrete_data, est = p.apply(h)
info = p.info

bn = Networks.HybridBN(use_mixture=False, has_logit=True)  # all may vary
bn.add_nodes(descriptor=info)
bn.add_edges(data=discrete_data, optimizer='HC', scoring_function=('MI',), classifier=RandomForestClassifier())

bn.get_info(as_df=False)
t1 = time.time()
bn.fit_parameters(data=h)
t2 = time.time()
print(f'PL elaspsed: {t2 - t1}')

bn.get_params_tree("final.json")

# # bn.plot('Hybrid_hackp')
# for num, el in enumerate(bn.sample(10, as_df=False), 1):
#     print('\n', num)
#     for name, val in el.items():
#         print(f"{name: <15}", val)

# print('id'.ljust(5), list(bn.distributions.keys())[0:20])
# for num, el in enumerate(bn.sample(100, as_df=False), 1):
#     print(f"{num: <5}", [el[key] for key in list(bn.distributions.keys())[0:20]])
