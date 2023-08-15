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

    from bamt.networks.discrete_bn import DiscreteBN

    bn = DiscreteBN()

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
               # blacklist edges, list of tuples (node1, node2)
               'bl_add':[...]
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
