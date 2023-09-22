Learn and Save Bayesian Network
===============================

Used imports:

.. code-block:: python

    from bamt.preprocessors import Preprocessor
    import pandas as pd
    from sklearn import preprocessing as pp
    from bamt.networks import HybridBN


Let's start with data loading and preprocessing: 


.. code-block:: python

    data = pd.read_csv("data/real data/hack_processed_with_rf.csv")[
    ['Tectonic regime', 'Period', 'Lithology', 'Structural setting',
     'Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth']]

    # set encoder and discretizer
    encoder = pp.LabelEncoder()
    discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    # create preprocessor object with encoder and discretizer
    p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])

    # discretize data for structure learning
    discretized_data, est = p.apply(data)

    # get information about data
    info = p.info

Then we create a network object and perform structure and parameters learning:

.. code-block:: python

    # initialize network object
    bn = HybridBN(use_mixture=True, has_logit=True)

    # add nodes to network
    bn.add_nodes(info)
 
    # using mutual information as scoring function for structure learning
    bn.add_edges(discretized_data, scoring_function=('MI',))

    # or use evolutionary algorithm to learn structure

    bn.add_edges(discretized_data, optimizer = 'evo')

    bn.fit_parameters(data)

To save structure and parameters of the network separately, we can use the following code:

.. code-block:: python

    # saving structure
    bn.save_structure("hack_structure.json")
    # saving parameters
    bn.save_params("hack_p.json")

Or, if we want to save the whole network, we can use:

.. code-block:: python
    
    bn.save("hack_network.json")
