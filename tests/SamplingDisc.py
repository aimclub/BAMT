import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)



from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
from bamt import Networks
from pgmpy.estimators import K2Score

data = pd.read_csv("data/hack_processed_with_rf.csv")
data = data[['Tectonic regime', 'Period', 'Lithology', 'Structural setting']]
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)


encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])


discretized_data, est = p.apply(data)
info = p.info


bn = Networks.DiscreteBN()
bn.add_nodes(descriptor=info)

params = {'init_nodes': None,
          'bl_add': None,
          'cont_disc': None}

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('K2', K2Score), params=params)
bn.fit_parameters(data=data)
# for i in range(1000):
#     print(bn.nodes[0].choose(bn.distributions['Tectonic regime']))
print(bn.sample(1000))
