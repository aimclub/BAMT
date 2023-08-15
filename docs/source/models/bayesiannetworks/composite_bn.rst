Composite Bayesian Networks
------------------------

.. autoclass:: bamt.networks.composite_bn.CompositeBN
   :members:
   :no-undoc-members:

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If the dataset contains both discrete and continuous variables, ``CompositeBN`` is can be used.
To initialize a ``CompositeBN`` object, you can use the following code:

.. code-block:: python

    from  bamt.networks.composite_bn import CompositeBN

    bn = CompositeBN()


Data Preprocessing
~~~~~~~~~~~~~~~~~~

Before applying any structure or parametric learning, the data should be preprocessed as follows:

.. code-block:: python

    import bamt.Preprocessor as pp
    import pandas as pd
    from sklearn import preprocessing

    data = pd.read_csv("path/to/data")
    encoder = preprocessing.LabelEncoder()
    p = pp.Preprocessor([("encoder", encoder)])

    preprocessed_data, _ = p.apply(data)



Structure Learning
~~~~~~~~~~~~~~~~~~

For structure learning of Composite BNs, ``bn.add_nodes()`` and ``bn.add_edges()`` methods are used.
Data should be non-preprocessed when passed to ``bn.add_edges()``

.. code-block:: python

    info = p.info

    bn.add_nodes(info)

    bn.add_edges(data) # !!! non-preprocessed


Parametric Learning
~~~~~~~~~~~~~~~~~~~

For parametric learning of continuous BNs, ``bn.fit_parameters()`` method is used.

.. code-block:: python

    bn.fit_parameters(data) # !!! non-preprocessed
    bn.get_info()
