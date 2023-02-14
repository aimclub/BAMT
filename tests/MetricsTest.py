import bamt.networks as Networks
from sklearn import preprocessing as pp
import pandas as pd
from bamt.preprocessors import Preprocessor
import time

start = time.time()


p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("data/real data/hack_processed_with_rf.csv")
cols = [
    'Tectonic regime',
    'Period',
    'Lithology',
    'Structural setting',
    'Gross',
    'Netpay',
    'Porosity',
    'Permeability',
    'Depth']
h = h[cols]

print(h.describe())
print("-----")
p2 = time.time()
print(f"Time elapsed for preparing data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(
    n_bins=5,
    encode='ordinal',
    strategy='quantile')

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

columns = ['Lithology', 'Structural setting', 'Porosity', 'Depth']
validY = h[columns].dropna()
validX = h.drop(columns, axis=1).dropna()

time_1 = time.time()
pred_param = bn.predict(validX, parall_count=3)
time_2 = time.time()
print(pred_param)
print(f'Predict elapsed: {time_2 - time_1}')
