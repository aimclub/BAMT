.. BAMT documentation master file, created by
   sphinx-quickstart on Thu Jan 19 21:13:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BAMT's documentation!
================================

The site contains documentation for the `BAMT framework <https://github.com/ITMO-NSS-team/BAMT>`__.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   self
   getting_started/install.rst
   getting_started/contribution.rst
   getting_started/faq.rst
   getting_started/cite_us.rst


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API

   api/builders.rst
   api/mi_entropy_gauss.rst
   api/networks.rst
   api/nodes.rst
   api/preprocessors.rst
   api/redef_HC.rst
   api/redef_info_scores.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Models

   models/BayesianNetworks.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: BN Algorithms

   bnalgs/bn_learning.rst
   bnalgs/big_bns.rst
   bnalgs/applied_tasks.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Examples, workflow pipelines

    examples/learn_save.rst
    examples/learn_params_vis.rst
    examples/read_structure_param_learning.rst
    examples/read_structure_and_params.rst
    examples/learn_sampling_predict.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
