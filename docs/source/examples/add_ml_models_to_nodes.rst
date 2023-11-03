Adding Machine Learning models to Bayesian Network nodes
========================================================

BAMT supports adding machine learning models to Bayesian Network nodes.

First, lets import BAMT modules and required machine learning modules.

.. code-block:: python

    import bamt.networks as networks
    import bamt.preprocessors as pp

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble.RandomForestRegressor import RandomForestRegressor
    from sklearn.linear_model.LinearRegression import LinearRegression

    from pgmpy.estimators import K2Score

Let's start with data importing and preprocessing.

.. code-block:: python

    # Importing data
    data = pd.read_csv(r'../Data/real data/vk_data.csv')

    # Choose columns
    cols = ['age',
            'sex',
            'has_pets',
            'is_parent',
            'relation',
            'is_driver',
            'tr_per_month',
            'median_tr',
            'mean_tr']
    data = data[cols]
    data[['sex',
          'has_pets',
          'is_parent',
          'relation',
          'is_driver']] = data[['sex',
                                'has_pets',
                                'is_parent',
                                'relation',
                                'is_driver']].astype(str)

    # Preprocessing

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    info = p.info
    info

Next, we initialize Bayesian Network object and add nodes to it.

.. code-block:: python

    bn = networks.HybridBN(has_logit=True, use_mixture=True)
    bn.add_nodes(info)

After adding nodes we can perform structure learning.

.. code-block:: python

    bn.add_edges(discretized_data,  scoring_function=('K2',K2Score))

Finally, before parameters learning, we can add machine learning models to nodes.
Let's add classifier models to discrete nodes and regressor models to continuous nodes and perform parameters learning.

.. code-block:: python

    bn.set_classifiers(classifiers={'age': DecisionTreeClassifier(),
                             'relation': RandomForestClassifier(),
                             'is_driver': KNeighborsClassifier(n_neighbors=2)})
    bn.set_regressors(regressors={'tr_per_month': RandomForestRegressor(),
                                    'mean_tr': LinearRegression()})

    bn.fit_parameters(data)

Now, we can save the model to load it later.

.. code-block:: python

    bn.save('vk_model.json')
    bn.load('vk_model.json')

Or visualize it (the html won't be rendered in jupyter notebook, but it will be rendered in html file and saved):

.. code-block:: python

    bn.plot('vk_model.html')
