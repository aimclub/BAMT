Bayesian Networks
=================

BaseNetwork class, Hill Climbing and Evolutionary Algorithms
------------------------------------------------------------

BaseNetwork class
~~~~~~~~~~~~~~~~~

All three BN types are based on an abstract class ``BaseNetwork``. 
This class provides the basic functions for all BN types.
The three BN types are ``DiscreteBN``, ``ContinuousBN`` and ``HybridBN``.
The ``HybridBN`` is a BN that contains both discrete and continuous variables.
The ``DiscreteBN`` and ``ContinuousBN`` are two BN types that are used to represent the BNs that contain only discrete or continuous variables, respectively.

.. autoclass:: bamt.networks.BaseNetwork
   :members:
   :no-undoc-members:

Hill Climbing and Evolutionary Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently BAMT employs Hill Climbing and Evolutionary Algorithms to learn the structure of the BNs. To use them,
you need to specify the ``optimizer`` parameter in ``add_edges`` method. Here is an example:

For Example:

.. code-block:: python

    from bamt.networks.discrete_bn import DiscreteBN
    import bamt.preprocessors as pp
    import pandas as pd

    asia = pd.read_csv('data.csv')
    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(asia)

    bn = DiscreteBN()
    info = p.info
    info

    # add edges using Hill Climbing
    bn.add_edges(discretized_data, optimizer='HC')
    # add edges using Evolutionary Algorithm
    bn.add_edges(discretized_data, optimizer='Evo')



Evolutionary Algorithm has these additional parameters:

        :param data: The data from which to build the structure.
        :type data: DataFrame
        :param classifier: A classification model for discrete nodes, defaults to None.
        :type classifier: Optional[object]
        :param regressor: A regression model for continuous nodes, defaults to None.
        :type regressor: Optional[object]

        :Keyword Args:
            * *init_nodes* (list) -- Initial nodes to be included in the population.
            * *max_arity* (int) -- Maximum arity for the evolutionary algorithm.
            * *timeout* (int) -- Timeout for the evolutionary algorithm in minutes.
            * *pop_size* (int) -- Population size for the evolutionary algorithm.
            * *crossover_prob* (float) -- Crossover probability for the evolutionary algorithm.
            * *mutation_prob* (float) -- Mutation probability for the evolutionary algorithm.
            * *custom_mutations* (list) -- Custom mutation types for the evolutionary algorithm.
            * *custom_crossovers* (list) -- Custom crossover types for the evolutionary algorithm.
            * *selection_type* (SelectionTypesEnum) -- Selection type for the evolutionary algorithm.
            * *blacklist* (list) -- Blacklist for the evolutionary algorithm.
            * *whitelist* (list) -- Whitelist for the evolutionary algorithm.
            * *custom_constraints* (list) -- Custom constraints for the evolutionary algorithm.
            * *custom_metric* (function) -- Custom objective metric for the evolutionary algorithm.

        The resulting structure is stored in the `skeleton` attribute of the `EvoStructureBuilder` object.

HillClimbing parameters are described below in DiscreteBN, ContinuousBN and HybridBN sections.