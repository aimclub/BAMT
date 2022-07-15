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

bn = Networks.HybridBN(use_mixture=True, has_logit=True)
info = p.info

bn.add_nodes(info)

structure = [("Tectonic regime", "Structural setting"),
             ("Gross", "Netpay"),
             ("Lithology", "Permeability")]

bn.set_structure(edges=structure)

bn.get_info(as_df=False)

with open("hack_p.json") as params:
    params = json.load(params)
    bn.set_parameters(params)

bn.plot("gg3.html")

bn2 = Networks.HybridBN(use_mixture=True, has_logit=True)
bn2.load("hack.json")

print(bn2.sample(50))
