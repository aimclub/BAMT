from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import bamt.Networks as Networks
import json

hack_data = pd.read_csv("data/real data/hack_processed_with_rf.csv")[
    ['Tectonic regime', 'Period', 'Lithology', 'Structural setting',
     'Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']]

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

discretized_data, est = p.apply(hack_data)
info = p.info

bn2 = Networks.HybridBN(use_mixture=True)
bn2.add_nodes(info)

with open("hack_p.json") as params:
    with open("hack_s.json") as structure:
        edges = json.load(structure)
        params = json.load(params)
        bn2.set_structure(edges=edges)

# bn2.get_info(as_df=False)
# bn2.plot("gg2.html")
