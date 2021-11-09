from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks
from Nodes import BaseNode, MixtureGaussianNode

# from pgmpy.estimators import K2Score

vk_data = pd.read_csv("../Data/vk_data.csv")
ROWS = 50
vk_data = vk_data.iloc[:ROWS, :]

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

nodes_type_mixed = p.get_nodes_types(vk_data)

columns = [col for col in vk_data.columns.to_list() if nodes_type_mixed[col] in ['disc', 'disc_num']]
discrete_data = vk_data[columns]

discretized_data, est = p.apply(discrete_data)
info = p.info

info = {"types": p.get_nodes_types(discretized_data),
        "signs": info['signs']}

# Make some errors
info['types']['relation'] = 'unknown'
info['types']['sex'] = 'third'

print(info['signs'])

bn = Networks.DiscreteBN()
bn.add_nodes(descriptor=info)

# ----------

bn.set_nodes(relation="A node")


class MyNode():
    pass

    def __repr__(self):
        return 'MyNode'


bn.set_nodes(relation=MyNode)

bn.set_nodes(relation=MixtureGaussianNode)
print(bn.nodes[-5:])

class CorrectNode(BaseNode):
    pass

bn.set_nodes(sex=CorrectNode)

print(bn.nodes[-5:])
