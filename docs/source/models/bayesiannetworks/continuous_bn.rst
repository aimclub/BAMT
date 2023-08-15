Continuous Bayesian Networks
----------------------------

.. autoclass:: bamt.networks.continuous_bn.ContinuousBN
   :members:
   :no-undoc-members:

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If all the variables in dataset are continuous, ``ContinuousBN`` is recommended to use. 
To initialize a ``ContinuousBN`` object, you can use the following code:

.. code-block:: python

    from bamt.networks.continuous_bn import ContinuousBN
    
    bn = ContinuousBN(use_mixture=True)

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
               # blacklist edges, list of tuples (node1, node2)
               'bl_add':[...]
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