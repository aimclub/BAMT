from bamt.preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import bamt.networks as Networks
# import json

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

bn.fit_parameters(data=hack_data)
print(bn.sample(4))
bn.save_params("hack_p.json")
# bn.save_structure("hack_s.json")
bn.save("hack.json")
