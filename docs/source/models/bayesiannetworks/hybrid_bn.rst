Hybrid Bayesian Networks
------------------------

.. autoclass:: bamt.networks.hybrid_bn.HybridBN
   :members:
   :no-undoc-members:

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If the dataset contains both discrete and continuous variables, ``HybridBN`` is recommended to use.
To initialize a ``HybridBN`` object, you can use the following code:

.. code-block:: python

    from  bamt.networks.hybrid_bn import HybridBN
    
    bn = HybridBN(has_logit=True, use_mixture=True)

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

For parametric learning of continuous BNs, ``bn.fit_parameters()`` method is used. 

.. code-block:: python

    bn.fit_parameters(data)

    bn.get_info() # get information table about the network
