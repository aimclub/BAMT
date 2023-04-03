Learn and Visualize Bayesian Network
====================================

Used imports:

.. code-block:: python

    from bamt.networks.hybrid_bn import HybridBN
    import bamt.Preprocessors as pp

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    from pgmpy.estimators import K2Score

Importing example data:

.. code-block:: python

    data = pd.read_csv(r'../Data/real data/vk_data.csv')
    data

Choosing a chunk of data:

.. code-block:: python

    cols = ['age', 'sex', 'has_pets', 'is_parent', 'relation', 'is_driver', 'tr_per_month', 'median_tr', 'mean_tr']
    data = data[cols]
    data[['sex', 'has_pets',  'is_parent', 'relation', 'is_driver']] = data[['sex',	'has_pets',	'is_parent', 'relation', 'is_driver']].astype(str)

Preprocessing data, encode categorical features and discretize numerical features, initialize BN and add nodes:

.. code-block:: python 

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    bn = HybridBN(has_logit=True, use_mixture=True) # init BN
    info = p.info
    info
    bn.add_nodes(info)

Learning BN structure and parameters with HillClimbing algorithm:

.. code-block:: python 


    bn.add_edges(discretized_data,  scoring_function=('K2',K2Score))
    bn.set_classifiers(classifiers={'age': DecisionTreeClassifier(),
                                 'relation': RandomForestClassifier(),
                                 'is_driver': KNeighborsClassifier(n_neighbors=2)})
    bn.fit_parameters(data)

Visualize BN structure:

.. code-block:: python 

    bn.plot('bn.html')

The visualized BN structure will not be rendered by jupyter notebook, but you can see it in the root directory of the project.
