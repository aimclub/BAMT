Bayesian Networks
=================

BaseNetwork class
-----------------

All three BN types are based on an abstract class ``BaseNetwork``. 
This class provides the basic functions for all BN types.
The three BN types are ``DiscreteBN``, ``ContinuousBN`` and ``HybridBN``.
The ``HybridBN`` is a BN that contains both discrete and continuous variables.
The ``DiscreteBN`` and ``ContinuousBN`` are two BN types that are used to represent the BNs that contain only discrete or continuous variables, respectively.

.. autoclass:: bamt.networks.BaseNetwork
   :members:
   :no-undoc-members: