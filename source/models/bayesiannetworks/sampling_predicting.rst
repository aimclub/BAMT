Sampling and Prediction with Bayesian Networks
----------------------------------------------

Sampling with Bayesian Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For data sampling from any BNs, ``bn.sample()`` method is used, but the network should be parametrically fitted first.

.. code-block:: python

    bn.fit_parameters(data)
    sampled_data = bn.sample(1000) # sample 1000 data points



Predicting with Bayesian Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For prediction with any BNs, ``bn.predict()`` method is used, but the network should be also parametrically fitted first.

.. code-block:: python

    bn.fit_parameters(data_train)

    # parall_count is the number of parallel threads to use
    predictions = bn.predict(test=data_test, parall_count=4) 
