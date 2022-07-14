from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import bamt.Networks as Networks
from bamt.Nodes import BaseNode, MixtureGaussianNode
import bamt.Nodes as nodes

vk_data = pd.read_csv("../Data/real data/vk_data.csv").iloc[0:50]

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

discretized_data, est = p.apply(vk_data)
info = p.info

# Make some errors
info['types']['relation'] = 'unknown'
info['types']['sex'] = 'helicopter'

bn = Networks.HybridBN()
bn.add_nodes(descriptor=info)

# ----------

bn.set_nodes(['A node'])
print(bn.nodes == [])


class MyNode:
    pass

    def __repr__(self):
        return 'MyNode'


bn.set_nodes([MyNode])

bn.set_nodes([MixtureGaussianNode])
print(bn.nodes == [])


class CorrectNode(BaseNode):
    type = "Gaussian"
    pass


bn.set_nodes([CorrectNode(name='Node3'),
              nodes.DiscreteNode(name='Node6')],
             info={"types": {"Node3": "cont", "Node6": "disc"},
                   "signs": {"Node3": "neg", "Node6": "pos"}})
print(bn.nodes != [])

# ----------

bn.set_edges([["maybe a node", "definitely not a node"], ["imposter1", "imposter2"]])
print(bn.edges == [])

bn.set_edges([['Node3', 'Node6']])
print(bn.edges != [])

# --------

ns = []
for d, g in zip(
        [nodes.GaussianNode(name="Node" + str(id)) for id in range(0, 3)],
        [nodes.DiscreteNode(name="Node" + str(id)) for id in range(3, 6)]):
    ns.append(d)
    ns.append(g)

es = []

for i in range(len(ns) - 1):
    es.append([ns[i].name, ns[i + 1].name])

t = {"Node" + str(i): "cont" for i in range(0, 3)}
for i in range(3, 6):
    t[f"Node{i}"] = "disc"
bn.set_structure(nodes=ns, edges=es,
                 info={"types": t,
                       "signs": {"Node" + str(i): "pos" for i in range(0, 6)}})

bn.plot("gg.html")
