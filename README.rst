.. image:: /docs/images/BAMT_white_bg.png
   :align: center
   :alt: BAMT framework logo

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi| |py_8| |py_9| |py_10|
   * - docs
     - |docs|
   * - license
     - | |license|
   * - stats
     - | |downloads_stats| |downloads_monthly| |downloads_weekly|

Repository of a data modeling and analysis tool based on Bayesian networks

BAMT - Bayesian Analytical and Modelling Toolkit. This repository contains a data modeling and analysis tool based on Bayesian networks. It can be divided into two main parts - algorithms for constructing and training Bayesian networks on data and algorithms for applying Bayesian networks for filling gaps, generating synthetic data, assessing edges strength e.t.c.

Installation
^^^^^^^^^^^^

BAMT package is available via PyPi:

.. code-block:: bash

   pip install bamt

BAMT Features
^^^^^^^^^^^^^

The following algorithms for Bayesian Networks learning are implemented:


* Building the structure of a Bayesian network based on expert knowledge by directly specifying the structure of the network;
* Building the structure of a Bayesian network on data using three algorithms - Hill Climbing, evolutionary and PC (evolutionary and PC are currently under development). For Hill Climbing, the following score functions are implemented - MI, K2, BIC, AIC. The algorithms work on both discrete and mixed data.
* Learning the parameters of distributions in the nodes of the network based on Gaussian distribution and Mixture Gaussian distribution with automatic selection of the number of components. 
* Non-parametric learning of distributions at nodes using classification and regression models. 
* BigBraveBN - algorithm for structural learning of Bayesian networks with a large number of nodes. Tested on networks with up to 500 nodes.

Difference from existing implementations:


* Algorithms work on mixed data;
* Structural learning implements score-functions for mixed data;
* Parametric learning implements the use of a mixture of Gaussian distributions to approximate continuous distributions;
* Non-parametric learning of distributions with various user-specified regression and classification models;
* The algorithm for structural training of large Bayesian networks (> 10 nodes) is based on local training of small networks with their subsequent algorithmic connection.

  .. image:: img/BN_gif.gif
     :target: img/BN_gif.gif
     :alt: title

For example, in terms of data analysis and modeling using Bayesian networks, a pipeline has been implemented to generate synthetic data by sampling from Bayesian networks.

.. image:: img/synth_gen.png
   :target: img/synth_gen.png
   :alt: title


How to use
^^^^^^^^^^

Then the necessary classes are imported from the library:

.. code-block:: python

   import bamt.networks as Nets

Next, a network instance is created and training (structure and parameters) is performed:

.. code-block:: python

   bn = Nets.HybridBN(has_logit=False, use_mixture=True)
   bn.add_edges(discretized_data,  scoring_function=('K2',K2Score))
   bn.fit_parameters(data)



Examples & Tutorials
^^^^^^^^^^^^^^^^^^^^^^

More examples can be found in `tutorials <https://github.com/ITMO-NSS-team/BAMT/tree/master/tutorials>`__  and `Documentation <https://bamt.readthedocs.io/en/latest/examples/learn_save.html>` .

Publications about BAMT
^^^^^^^^^^^^^^^^^^^^^^^



Project structure
^^^^^^^^^^^^^^^^^

The latest stable version of the library is available in the master branch.

It includes the following modules and direcotries:

* `bamt <https://github.com/ITMO-NSS-team/BAMT/tree/master/bamt>`__ - directory with the framework code:
    * Preprocessing - module for data preprocessing
    * Networks - module for building and training Bayesian networks
    * Nodes - module for nodes support of Bayesian networks
    * Utilities - module for mathematical and graph utilities
* `data <https://github.com/ITMO-NSS-team/BAMT/tree/master/data>`__  - directory with data for experiments and tests
* `tests <https://github.com/ITMO-NSS-team/BAMT/tree/master/tests>`__  - directory with unit and integration tests
* `tutorials <https://github.com/ITMO-NSS-team/BAMT/tree/master/tutorials>`__  - directory with tutorials
* `docs <https://github.com/ITMO-NSS-team/BAMT/tree/master/docs>`__ - directory with RTD documentation

Preprocessing
=============

Preprocessor module allows user to transform data according pipeline (similar to pipeline in scikit-learn).

Networks
========

Three types of networks are implemented:

* HybridBN - Bayesian network with mixed data
* DiscreteBN - Bayesian network with discrete data
* ContinuousBN - Bayesian network with continuous data

They are inherited from the abstract class BaseNetwork.

Nodes
=====

Contains classes for nodes of Bayesian networks.

Utilities
=========

Utilities module contains mathematical and graph utilities to support the main functionality of the library.


Web-BAMT
^^^^^^^^

A web interface for BAMT is currently under development. 
The repository is available at `web-BAMT <https://github.com/aimclub/Web-BAMT>`__ 

Contacts
^^^^^^^^

If you have questions or suggestions, you can contact us at the following address: ideeva@itmo.ru (Irina Deeva)

Citation
^^^^^^^^

@misc{BAMT,
author={BAMT},
title = {Repository experiments and data},
year = {2021},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/ITMO-NSS-team/BAMT.git}},
url = {https://github.com/ITMO-NSS-team/BAMT.git}
}

@article{deeva2023advanced,
title={Advanced Approach for Distributions Parameters Learning in Bayesian Networks with Gaussian Mixture Models and Discriminative Models},
author={Deeva, Irina and Bubnova, Anna and Kalyuzhnaya, Anna V},
journal={Mathematics},
volume={11},
number={2},
pages={343},
year={2023},
publisher={MDPI}
}

@article{kaminsky2022bigbravebn,
title={BigBraveBN: algorithm of structural learning for bayesian networks with a large number of nodes},
author={Kaminsky, Yury and Deeva, Irina},
journal={Procedia Computer Science},
volume={212},
pages={191--200},
year={2022},
publisher={Elsevier}
}


.. |docs| image:: https://readthedocs.org/projects/bamt/badge/?version=latest
    :target: https://bamt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |pypi| image:: https://badge.fury.io/py/bamt.svg
    :target: https://badge.fury.io/py/bamt

.. |py_10| image:: https://img.shields.io/badge/python_3.10-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.10-passing-success

.. |py_8| image:: https://img.shields.io/badge/python_3.8-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.8-passing-success

.. |py_9| image:: https://img.shields.io/badge/python_3.9-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.9-passing-success

.. |license| image:: https://img.shields.io/github/license/ITMO-NSS-team/BAMT
   :alt: Supported Python Versions
   :target: https://github.com/ITMO-NSS-team/BAMT/blob/master/LICENCE

.. |downloads_stats| image:: https://static.pepy.tech/personalized-badge/bamt?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads
 :target: https://pepy.tech/project/bamt
 
.. |downloads_monthly| image:: https://static.pepy.tech/personalized-badge/bamt?period=month&units=international_system&left_color=grey&right_color=blue&left_text=downloads/month
 :target: https://pepy.tech/project/bamt

.. |downloads_weekly| image:: https://static.pepy.tech/personalized-badge/bamt?period=week&units=international_system&left_color=grey&right_color=blue&left_text=downloads/week
 :target: https://pepy.tech/project/bamt
