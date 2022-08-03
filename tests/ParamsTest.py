from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import bamt.Networks as Networks

vk_data = pd.read_csv(r"data\real data\vk_data.csv").sample(150)

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder)])
discretized_data, est = p.apply(vk_data)
info = p.info

bn = Networks.HybridBN(has_logit=False, use_mixture=False)
bn.add_nodes(descriptor=info)
params = {"init_nodes": ["sex", "has_pets", "is_parent", "relation", "tr_per_month"],
          "init_edges": [("age", "mean_tr"), ("sex", "mean_tr"), ("sex", "has_pets"),
                         ("is_parent", "has_pets"), ("has_pets", "median_tr"),
                         ("is_driver", "tr_per_month"), ("tr_per_month", "median_tr"),
                         ("tr_per_month", "relation")]}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',), params=params)
bn.fit_parameters(data=vk_data)

# bn.get_info(as_df=False)
bn.sample(n=100)
