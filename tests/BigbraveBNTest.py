import os,sys
from pathlib import Path
currentdir = Path(os.getcwd())
sys.path.insert(0, os.path.dirname(currentdir.parent))
sys.path.insert(0, os.path.dirname(currentdir))
print(str(sys.path[0:3]))

import bamt.Networks as Nets
import bamt.Preprocessors as pp
from bamt.Networks import BigBraveBN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pgmpy.estimators import K2Score
from sklearn import preprocessing

data_discrete = pd.read_csv(r"../Data/benchmark/pigs.csv")
data_continuous = pd.read_csv(r"../Data/benchmark/arth150.csv")

encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(data_discrete)

info = p.info

space_restrictor = BigBraveBN()

space_restrictor.get_possible_edges_by_brave(
    df = data_discrete)

ps = space_restrictor.possible_edges

bn_discrete = Nets.DiscreteBN()

bn_discrete.add_nodes(descriptor=info)

params = {'white_list': ps}
bn_discrete.add_edges(discretized_data, scoring_function=('K2',K2Score), params=params)

encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(data_continuous)

info = p.info

space_restrictor = BigBraveBN()

space_restrictor.get_possible_edges_by_brave(
    df = data_continuous)

ps = space_restrictor.possible_edges

bn_continuous = Nets.ContinuousBN()

bn_continuous.add_nodes(descriptor=info)

params = {'white_list': ps}
bn_continuous.add_edges(discretized_data, scoring_function=('K2',K2Score), params=params)
