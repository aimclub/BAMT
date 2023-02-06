Read BN Structure from a File, Learn Distribution Parameters
============================================================

Used imports:

.. code-block:: python

    from bamt.preprocessors import Preprocessor
    import pandas as pd
    from sklearn import preprocessing as pp
    import bamt.networks as Networks
    import json

There are two options for loading a BN structure. The first is to read it directly from a JSON file:


.. code-block:: python

    bn = Networks.HybridBN(use_mixture=True, has_logit=True)

    bn2.load("structure.json")


The second one is to set it maually using list of edges:

.. code-block:: python

    structure = [("Tectonic regime", "Structural setting"),
                ("Gross", "Netpay"),
                ("Lithology", "Permeability")]

    bn.set_structure(edges=structure)

The next step is to learn parameters from data, to do this we need to read the data:

.. code-block:: python
    
    # reading data
    data = pd.read_csv("data.csv")

    # parameters learning
    bn.fit_parameters(data)
