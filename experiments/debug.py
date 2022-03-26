import sys
parentdir = 'C:\\Users\\anaxa\\Documents\\Projects\\BAMT'
sys.path.insert(0,parentdir) 



import pandas as pd
from sklearn import preprocessing
import seaborn as sns


from bamt.Preprocessors import Preprocessor
from bamt.ScoringFunctions import LLGMM, BICGMM
import bamt.Networks as Nets
from pgmpy.estimators import K2Score
from pgmpy.estimators import HillClimbSearch
import networkx as nx



data = pd.read_csv('Data/hack_processed_with_rf.csv')
cols = ['Gross', 'Netpay', 'Permeability', 'Porosity', 'Depth']
data = data[cols]

data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)


discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

p = Preprocessor([('discretizer', discretizer)])
discretized_data, est = p.apply(data)


bn = Nets.ContinuousBN()
info = p.info
bn.add_nodes(info)
bn.add_edges(data, scoring_function=('BIC',))