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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
