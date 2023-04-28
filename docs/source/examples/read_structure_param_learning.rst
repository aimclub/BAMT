Read BN Structure from a File, Learn Distribution Parameters
============================================================

Used imports:

.. code-block:: python

    from bamt.preprocessors import Preprocessor
    import pandas as pd
    from sklearn import preprocessing as pp
    from bamt.networks.hybrid_bn import HybridBN
    import json

There are two options for loading a BN structure. The first is to read it directly from a JSON file:


.. code-block:: python

    bn = HybridBN(use_mixture=True, has_logit=True)

    bn2.load("structure.json")


The second one is to set it manually using list of edges, but first nodes should be added:

.. code-block:: python

    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    
    info = p.info

    bn.add_nodes(info)

    structure = [("Tectonic regime", "Structural setting"),
                ("Gross", "Netpay"),
                ("Lithology", "Permeability")]

    bn.set_structure(edges=structure)

The next step is to learn parameters from data, to do this we need to read the data and perform parameters learning:

.. code-block:: python
    
    # reading data
    data = pd.read_csv("data.csv")

    # parameters learning
    bn.fit_parameters(data)
