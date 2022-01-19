import os
import sys

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, path)
# ---------
import time

start = time.time()

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks

# import Metrics

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

bn = Networks.HybridBN(has_logit=True)  # all may vary
bn.add_nodes(descriptor=info)
bn.add_edges(data=discrete_data, optimizer='HC', scoring_function=('MI',))

bn.get_info(as_df=False)
t1 = time.time()
bn.fit_parameters(data=h)
t2 = time.time()
print(f'PL elapsed: {t2 - t1}')

columns = ['Tectonic regime', 'Period', 'Gross']
validY = h[columns].dropna()
validX = h.drop(columns, axis=1).dropna()

time_1 = time.time()
pred_param = bn.predict(validX.iloc[0:20, :], parall_count=3)
time_2 = time.time()
print(f'Predict elapsed: {time_2 - time_1}')

# sync
# with matrix: 102.74379372596741
# with dict: 80.4413230419159, 82.58527779579163

# async
# Threads = 1
# Test for nan checker
# with for loop for columns: 88.78174304962158, 85.52139496803284
# with no for loop for columns: 94.38896083831787
# Threads = 2
# 55.83470320701599
# Threads = 3
# 53.80153179168701, 49.15917134284973

from sklearn.metrics import accuracy_score, mean_squared_error

for column in ['Tectonic regime', 'Period']:
    print(column, round(accuracy_score(validY[column][0:20], pred_param[column]), 5))

print('Gross', round(mean_squared_error(validY['Gross'][0:20], pred_param['Gross'], squared=False), 5))
