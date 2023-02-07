Sampling and Predicting
=======================

Used imports:

.. code-block:: python

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
    from sklearn.model_selection import train_test_split

    from pgmpy.estimators import K2Score

Importing exaple data:

.. code-block:: python

    data = pd.read_csv(r'../Data/real data/vk_data.csv')
    data

Choosing a chunk of data:

.. code-block:: python

    cols = ['age', 'sex', 'has_pets', 'is_parent', 'relation', 'is_driver', 'tr_per_month', 'median_tr', 'mean_tr']
    data = data[cols]
    data[['sex',	'has_pets',	'is_parent',	'relation',	'is_driver']] = data[['sex',	'has_pets',	'is_parent',	'relation',	'is_driver']].astype(str)

Prepocessing data, encode categorical features and discretize numerical features, initialize BN and add nodes:

.. code-block:: python 

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    bn = Nets.HybridBN(has_logit=True, use_mixture=True) # init BN
    info = p.info
    info
    bn.add_nodes(info)

Learinig BN structure and with HillClimbing algorithm:

.. code-block:: python 


    bn.add_edges(discretized_data,  scoring_function=('K2',K2Score))

Leatning distribution parameters and sampling data:

.. code-block:: python

    bn.fit_parameters(data)
    sampled_data = bn.sample(1000) # sample 1000 data points

Splitting data into train and test sets and droping target column:

.. code-block:: python 

    data_test, data_train = train_test_split(data, test_size=0.2, random_state=42)

    data_test = data_test.drop(columns=['tr_per_month'])

Predicting the target column with BN:

.. code-block:: python

    bn.fit_parameters(data_train)

    # parall_count is the number of parallel threads to use
    predictions = bn.predict(test=data_test, parall_count=4) 
