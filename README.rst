
BAMT
====

Repository of a data modeling and analysis tool based on Bayesian networks

BAMT - Bayesian Analytical and Modelling Toolkit. This repository contains a data modeling and analysis tool based on Bayesian networks. It can be divided into two main parts - algorithms for constructing and training Bayesian networks on data and algorithms for applying Bayesian networks for filling gaps, generating synthetic data, and searching for anomalous values.

Installation
^^^^^^^^^^^^

BAMT package is available via PyPi:

.. code-block:: bash

   pip install bamt

Bayesian network learning
=========================

In terms of training Bayesian networks on data, the following algorithms are implemented:


* Building the structure of a Bayesian network based on expert knowledge by directly specifying the structure of the network;
* Building the structure of a Bayesian network on data using three algorithms - Hill Climbing, evolutionary and PC. For Hill Climbing, the following score functions are implemented - MI, K2, BIC, AIC. The algorithms work on both discrete and mixed data.
* Training the parameters of distributions in the nodes of the network on the basis of data.

Difference from existing implementations:


* Algorithms work on mixed data;
* Structural learning implements score-functions for mixed data;
* Parametric learning implements the use of a mixture of Gaussian distributions to approximate continuous distributions;
* The algorithm for structural training of large Bayesian networks (> 10 nodes) is based on local training of small networks with their subsequent algorithmic connection.

  .. image:: img/BN_gif.gif
     :target: img/BN_gif.gif
     :alt: title

Generating synthetic data
=========================

In terms of data analysis and modeling using Bayesian networks, a pipeline has been implemented to generate synthetic data by sampling from Bayesian networks.


