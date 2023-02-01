.. _bayesiannetworks:

Bayesian Networks
=================

BaseNetwork class
-----------------

All three BN types are based on an abstract class ``BaseNetwork``. 
This class provides the basic functions for all BN types.
The three BN types are ``DiscreteBN``, ``ContinuousBN`` and ``HybridBN``.
The ``HybridBN`` is a BN that contains both discrete and continuous variables.
The ``DiscreteBN`` and ``ContinuousBN`` are two BN types that are used to represent the BNs that contain only discrete or continuous variables, respectively.

.. autoclass:: bamt.networks.BaseNetwork
   :members:
   :no-undoc-members:

Discrete Bayesian Networks
--------------------------

.. autoclass:: bamt.networks.DiscreteBN
   :members:
   :no-undoc-members:

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If all the variables in dataset are discrete, ``DiscreteBN`` is recommended to use. 
To initialize a ``DiscreteBN`` object, you can use the following code:

.. code-block:: python

    import bamt.Networks as Nets

    bn = Nets.DiscreteBN()

Data Preprocessing
~~~~~~~~~~~~~~~~~~

If the dataset contains ``float`` values (e.g. 1.0, 2.0 etc), they should be converted to ``integers`` or discretized before using ``DiscreteBN``.
Before applying any structure or parametric learning, the data should be preprocessed as follows:

.. code-block:: python

    import bamt.Preprocessor as pp
    import pandas as pd
    from sklearn import preprocessing

    data = pd.read_csv('data.csv')

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    info = p.info



Structure Learning
~~~~~~~~~~~~~~~~~~

For structure learning of discrete BNs, ``bn.add_nodes()`` and ``bn.add_edges()`` methods should be used.

.. code-block:: python

    from pgmpy.estimators import K2Score

    bn.add_nodes(info) # add nodes from info obtained from preprocessing

    bn.get_info() # to make sure that the network recognizes the variables as discrete

    params = {
                # Defines initial nodes of the network, list of node names
               'init_nodes':[...] 
                # Defines initial edges of the network, list of tuples (node1, node2)
               'init_edges':[...] 
                # Strictly set edges where algoritm must learn, list of tuples (node1, node2)
               'white_list':[...] 
                # Allow algorithm to remove edges defined by user, bool
               'remove_init_edges':True
              }

    # Structure learning using K2Score and parameters defined above
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params) 

    bn.plot('foo.html') # Plot the network, save it to foo.html, NOT rendered in notebook


Parametric Learning
~~~~~~~~~~~~~~~~~~~

For parametric learning of discrete BNs, ``bn.fit_parameters()`` method is used. 

.. code-block:: python

    bn.fit_parameters(data)

    bn.get_info() # get information table about the network



Continuous Bayesian Networks
----------------------------

.. autoclass:: bamt.networks.ContinuousBN
   :members:
   :no-undoc-members:

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If all the variables in dataset are continuous, ``ContinuousBN`` is recommended to use. 
To initialize a ``ContinuousBN`` object, you can use the following code:

.. code-block:: python

    import bamt.Networks as Nets
    
    bn = Nets.ContinuousBN(use_mixture=True)

ContinuousBN has an additional parameter ``use_mixture``. 
It is used to determine whether to use mixuters of Gaussian distributions  to represent the conditional distribution of continuous variables.
If ``use_mixture`` is ``True``, mixuters of Gaussian distributions are used to represent the conditional distribution of continuous variables. 


Data Preprocessing
~~~~~~~~~~~~~~~~~~

If the dataset contains ``integer`` values that should be treated as continuous variables (e.g. 1, 2 etc), they should be converted to ``float``.
Before applying any structure or parametric learning, the data should be preprocessed as follows:

.. code-block:: python

    import bamt.Preprocessor as pp
    import pandas as pd
    from sklearn import preprocessing

    data = pd.read_csv('data.csv')

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    info = p.info



Structure Learning
~~~~~~~~~~~~~~~~~~

For structure learning of continuous BNs, ``bn.add_nodes()`` and ``bn.add_edges()`` methods are used. 

.. code-block:: python

    from pgmpy.estimators import K2Score

    bn.add_nodes(info) # add nodes from info obtained from preprocessing

    bn.get_info() # to make sure that the network recognizes the variables as continuous

    params = {
                # Defines initial nodes of the network, list of node names
               'init_nodes':[...]
                # Defines initial edges of the network, list of tuples (node1, node2)
               'init_edges':[...]
                # Strictly set edges where algoritm must learn, list of tuples (node1, node2)
               'white_list':[...]
                # Allow algorithm to remove edges defined by user, bool
               'remove_init_edges':True
              }

    # Structure learning using K2Score and parameters defined above
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params) 

    bn.plot('foo.html') # add nodes from info obtained from preprocessing


Parametric Learning
~~~~~~~~~~~~~~~~~~~

For parametric learning of BNs, ``bn.fit_parameters()`` method is used. 

.. code-block:: python

    bn.fit_parameters(data)

    bn.get_info() # get information table about the network





Hybrid Bayesian Networks
------------------------

.. autoclass:: bamt.networks.HybridBN
   :members:
   :no-undoc-members:

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If the dataset contains both discrete and continuous variables, ``HybridBN`` is recommended to use.
To initialize a ``HybridBN`` object, you can use the following code:

.. code-block:: python

    import bamt.Networks as Nets
    
    bn = Nets.HybridBN(has_logit=True, use_mixture=True)

HybridBN has two additional parameters ``has_logit`` and ``use_mixture``.
``has_logit`` is used to determine whether to use logit nodes. Logit nodes use machine learning algorithms to represent variable.
Logit nodes are discrete nodes that have continuous root nodes; classification models are used to model conditional distributions in such nodes.



Data Preprocessing
~~~~~~~~~~~~~~~~~~

Before applying any structure or parametric learning, the data should be preprocessed as follows:

.. code-block:: python

    import bamt.Preprocessor as pp
    import pandas as pd
    from sklearn import preprocessing

    data = pd.read_csv('data.csv')

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    info = p.info



Structure Learning
~~~~~~~~~~~~~~~~~~

For structure learning of Hybrid BNs, ``bn.add_nodes()`` and ``bn.add_edges()`` methods are used. 

.. code-block:: python

    from pgmpy.estimators import K2Score

    bn.add_nodes(info) # add nodes from info obtained from preprocessing

    params = {
                # Defines initial nodes of the network, list of node names
               'init_nodes':[...]
                # Defines initial edges of the network, list of tuples (node1, node2)
               'init_edges':[...]
               # Strictly set edges where algoritm must learn, list of tuples (node1, node2)
               'white_list':[...]
               # Allow algorithm to remove edges defined by user, bool
               'remove_init_edges':True
              }

    # Structure learning using K2Score and parameters defined above
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params) 

    bn.plot('foo.html') # add nodes from info obtained from preprocessing


Parametric Learning
~~~~~~~~~~~~~~~~~~~

For parametric learning of continuous BNs, ``bn.fit_parameters()`` method is used. 

.. code-block:: python

    bn.fit_parameters(data)

    bn.get_info() # get information table about the network



Sampling and Prediction with Bayesian Networks
----------------------------------------------

Sampling with Bayesian Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For data sampling from any BNs, ``bn.sample()`` method is used, but the network should be parametrically fitted first.

.. code-block:: python

    bn.fit_parameters(data)
    sampled_data = bn.sample(1000) # sample 1000 data points



Predicing with Bayesian Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For prediction with any BNs, ``bn.predict()`` method is used, but the network should be also parametrically fitted first.

.. code-block:: python

    bn.fit_parameters(data_train)

    # parall_count is the number of parallel threads to use
    predictions = bn.predict(test=data_test, parall_count=4) 


Algorithms for Large Bayesian Networks
--------------------------------------


BigBraveBN
~~~~~~~~~~

BigBraveBN is an algorithm that is used for structure learning of large Bayesian networks.
It restricts the search space by using Brave coefficient, that represents mutual occurrence of two variables in groups.
These groups are formed for each variable using kNN algorithm that searches nearest neighbors for each variable.
Mutual information score is used as metric for nearest neighbors algorithm. 


.. autoclass:: bamt.networks.BigBraveBN
   :members:
   :no-undoc-members:



BigBraveBN initialization and usage
-----------------------------------


To use BigBraveBN, just follow typical structure learning procedure with one difference: use ``BigBraveBN`` to generate ``white_list``.

First, initialize ``BigBraveBN`` object and generate possible edges list:

.. code-block:: python

    space_restrictor = BigBraveBN()

    space_restrictor.set_possible_edges_by_brave(
      df = data)

    ps = space_restrictor.possible_edges

Then, preprocess the data: 

.. code-block:: python

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)

    info = p.info

Then perform structure learning as usual, but use ``ps`` as ``white_list``:

.. code-block:: python

    bn = Nets.ContinuousBN()

    bn.add_nodes(descriptor=info)

    params = {'white_list': ps}

    bn.add_edges(discretized_data, scoring_function=('K2',K2Score), params=params)
