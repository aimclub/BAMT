import sys
import os

path = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, path)

from Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import Networks

print('Import complete')

h = pd.read_csv("../Data/hack_processed_with_rf.csv")
cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross', 'Netpay', 'Porosity', 'Permeability',
        'Depth']
h = h[cols]

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

discrete_data, est = p.apply(h)
info = p.info

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifier_list = [
    None,
    KNeighborsClassifier(n_neighbors=3),
    RandomForestClassifier(),
    DecisionTreeClassifier()]

print('Init bn')
bn = Networks.HybridBN(use_mixture=False, has_logit=True)
bn.add_nodes(descriptor=info)
bn.add_edges(data=discrete_data, optimizer='HC', scoring_function=('MI',))

bn.set_classifiers(classifiers={'Structural setting': DecisionTreeClassifier(),
                                'Lithology': RandomForestClassifier(),
                                'Period': KNeighborsClassifier(n_neighbors=2)})
bn.get_info(as_df=False)

bn.fit_parameters(data=h)
print('fp finished')

predictions_mi = bn.sample(514, as_df=True)

bn.get_params_tree('final.json')
print('finished')

# successful = {}
# failed = {}
# for i in range(6):
#     successful[i] = []
#     failed[i] = []
#     for classifier in classifier_list:
#         # try:
#             bn = Networks.HybridBN(use_mixture=False, has_logit=True)  # all may vary
#             bn.add_nodes(descriptor=info)
#             bn.add_edges(data=discrete_data, optimizer='HC', scoring_function=('MI',), classifier=classifier)
#             bn.fit_parameters(data=h)
#
#             bn.sample(10, as_df=False)
#             successful[i].append(classifier)
#         # except Exception as ex:
#             # print(classifier, ex)
#             # failed[i].append(classifier)
# for i, result in successful.items():
#     print(i, result)
# for i, result in failed.items():
#     print(i, failed)
