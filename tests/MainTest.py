import pandas as pd
from sklearn import preprocessing as pp
from pgmpy.estimators import K2Score

from bamt.Preprocessors import Preprocessor
import bamt.Networks as Networks

'''
Optional:
You can also uncomment print() that you need.
'''

hack_data = pd.read_csv("data/real data/hack_processed_with_rf.csv")
cont_data = hack_data[['Gross', 'Netpay', 'Porosity',
                       'Permeability', 'Depth']]
disc_data = hack_data[['Tectonic regime', 'Period',
                       'Lithology', 'Structural setting']]
hybrid_data = hack_data[['Tectonic regime', 'Period',
                         'Lithology', 'Structural setting',
                         'Gross', 'Netpay', 'Porosity',
                         'Permeability', 'Depth']]

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5,
                                  encode='ordinal',
                                  strategy='uniform')
p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

# Discrete pipeline
discretized_data, _ = p.apply(disc_data)
disc_bn = Networks.DiscreteBN()
info = p.info
disc_bn.add_nodes(info)
disc_bn.add_edges(data=discretized_data, scoring_function=('K2', K2Score))
disc_bn.fit_parameters(data=disc_data)
synth_disc_data = disc_bn.sample(50)

disc_bn.save('./disc_bn.json')
disc_bn2 = Networks.DiscreteBN()
disc_bn2.load('./disc_bn.json')
synth_disc_data2 = disc_bn2.sample(50)
# print(disc_bn.get_info())
# print(disc_bn2.get_info())
# print(synth_disc_data)
# print(synth_disc_data2)

# Continuous pipeline
discretized_data, _ = p.apply(cont_data)
cont_bn = Networks.ContinuousBN(use_mixture=True)
info = p.info
cont_bn.add_nodes(info)
cont_bn.add_edges(data=discretized_data, scoring_function=('K2', K2Score))
cont_bn.fit_parameters(data=cont_data)
synth_cont_data = cont_bn.sample(50)

cont_bn.save('./cont_bn.json')
cont_bn2 = Networks.ContinuousBN(use_mixture=True)
cont_bn2.load('./cont_bn.json')
synth_cont_data2 = cont_bn2.sample(50)
# print(cont_bn.get_info())
# print(cont_bn2.get_info())
# print(synth_cont_data)
# print(synth_cont_data2)

# Hybrid pipeline
discretized_data, _ = p.apply(hybrid_data)
hybrid_bn = Networks.HybridBN(use_mixture=True)
info = p.info
hybrid_bn.add_nodes(info)
hybrid_bn.add_edges(data=discretized_data, scoring_function=('K2', K2Score))
hybrid_bn.fit_parameters(data=hybrid_data)
synth_hybrid_data = hybrid_bn.sample(50)

hybrid_bn.save('./hybrid_bn.json')
hybrid_bn2 = Networks.HybridBN(use_mixture=True)
hybrid_bn2.load('./hybrid_bn.json')
synth_hybrid_data2 = hybrid_bn2.sample(50)
# print(hybrid_bn.get_info())
# print(hybrid_bn2.get_info())
# print(synth_hybrid_data)
# print(synth_hybrid_data2)
