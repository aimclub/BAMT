Read Structure and Distribution Parameters from a File
======================================================

Used imports:

.. code-block:: python
    
    from bamt.preprocessors import Preprocessor
    import pandas as pd
    from sklearn import preprocessing as pp
    from bamt.networks import HybridBN
    import json

You can read the pre-trained structure and distribution parameters from a file.
This is useful if you do not want to wait for the structure learning every time you run the script or cell.

Here is an example of how to read structure and distribution parameters from a file:

.. code-block:: python

    bn = Networks.HybridBN(use_mixture=True, has_logit=True)

    bn.load("network_pretrained.json")

It is also possible to read structure and distribution parameters separately, if you saved them separately:

.. code-block:: python

    bn = HybridBN(use_mixture=True, has_logit=True)

    bn.load("network_pretrained_structure.json")
    bn.load("network_pretrained_distribution.json")
