

import bamt.Networks as Nets
import bamt.Preprocessors as pp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
from bamt.Builders import StructureBuilder

from pgmpy.estimators import K2Score
from gmr import GMM
import seaborn as sns

data = pd.read_csv(r'data/real data/hack_processed_with_rf.csv')
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
data = data[cols]
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
encoder = preprocessing.LabelEncoder()
discretizer = preprocessing.KBinsDiscretizer(
    n_bins=5, encode='ordinal', strategy='quantile')

p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(data)
bn = Nets.HybridBN(has_logit=False, use_mixture=False)  # init BN
info = p.info
bn.add_nodes(info)
bn.add_edges(discretized_data, scoring_function=('K2', K2Score))
bn.set_regressor(regressors={'Permeability': XGBRegressor()})
bn.fit_parameters(data)
test = dict(data.loc[1,
                     ['Tectonic regime',
                      'Period',
                      'Lithology',
                      'Structural setting',
                      'Gross',
                      'Netpay',
                      'Porosity',
                      'Depth']])
bn.sample(100)
