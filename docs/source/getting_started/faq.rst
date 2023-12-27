.. _faq:

FAQ
===
1. On ``calculate_weights`` I got the following:

.. code-block:: python

    assert np.all([A.dtype == "int" for A in Symbol_matrices])
    AssertionError

What should I do?

Answer:
    | Because of not so clear dtypes policies, ``pyitlib`` need all integer columns
    | as int type (col.dtype must return 'str'). So to fix you can do:
Instead of:

.. code-block:: python

    bn.calculate_weights(discretized_data)


Convert dtypes to intc:

.. code-block:: python

    bn.calculate_weights(discretized_data.astype(np.intc))


