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
import Metrics

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("../Data/hack_processed_with_rf.csv")
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

bn = Networks.HybridBN(use_mixture=True, has_logit=True)  # all may vary
bn.add_nodes(descriptor=info)
bn.add_edges(data=discrete_data, optimizer='HC', scoring_function=('MI',))

bn.get_info(as_df=False)
t1 = time.time()
bn.fit_parameters(data=h)
t2 = time.time()
print(f'PL elaspsed: {t2 - t1}')
accuracy_dict, rmse_dict, real_param, pred_param, indexes = Metrics.Accuracy().calculate_acc(bn=bn, data=h, columns=['Netpay'])
print(accuracy_dict, rmse_dict, real_param, pred_param, indexes)
