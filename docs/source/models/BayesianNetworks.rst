.. _bayesiannetworks:

Bayesian Networks
=================

All three BN types are based on an abstract class ``BaseNetwork``. 
This class provides the basic functions for all BN types.
The three BN types are ``DiscreteBN``, ``ContinuousBN`` and ``HybridBN``.
The ``HybridBN`` is a BN that contains both discrete and continuous variables.
The ``DiscreteBN`` and ``ContinuousBN`` are two BN types that are used to represent the BNs that contain only discrete or continuous variables, respectively.

.. autoclass:: bamt.Networks.BaseNetwork
   :members:
   :no-undoc-members:

Discrete Bayesian Networks
--------------------------

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If all the variables in dataset are discrete, ``DiscreteBN`` is recommended to use. 
To initialize a ``DiscreteBN`` object, you can use the following code:

.. code-block:: python

    import bamt.Networks as Nets

    bn = Nets.DiscreteBN()




Continuous Bayesian Networks
----------------------------

Network initialization
~~~~~~~~~~~~~~~~~~~~~~

If all the variables in dataset are continuous, ``ContinuousBN`` is recommended to use. 
To initialize a ``ContinuousBN`` object, you can use the following code:

.. code-block:: python

    import bamt.Networks as Nets
    
    bn = Nets.ContinuousBN(use_mixture=True)

ContinuousBN has an additional parameter ``use_mixture``. 
It is used to determine whether to use mixuters of Gaussian distributions  to represent the conditional distribution of continuous variables.
If ``use_mixture`` is ``True``, mixuters of Gaussian distributions are used to represent the conditional distribution of continuous variables. 


Hybrid Bayesian Networks
------------------------


Algorithms for Large Bayesian Networks
--------------------------------------