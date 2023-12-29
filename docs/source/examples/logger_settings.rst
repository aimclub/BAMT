Setting up loggers in BAMT
===============================

Used imports:

.. code-block:: python

    import pandas as pd

    from sklearn import preprocessing as pp

    import bamt.preprocessors as preprocessors
    from bamt.networks import ContinuousBN

    from bamt.log import bamt_logger

There are 2 methods to use: ``switch_console_out`` and ``switch_file_out`` of ``bamt_logger``.

By default, bamt will print out messages in console and will not use any log files.

How to turn off/on console output?
_______________________________

Let's consider this example:

.. code-block:: python

    def learn_bn():
        hack_data = pd.read_csv("data/real data/hack_processed_with_rf.csv")[
            [
                "Tectonic regime",
                "Period",
                "Lithology",
                "Structural setting",
                "Gross",
                "Netpay",
                "Porosity",
                "Permeability",
                "Depth",
            ]
        ].dropna()

        encoder = pp.LabelEncoder()
        discretizer = pp.KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")

        p = preprocessors.Preprocessor([("encoder", encoder), ("discretizer", discretizer)])

        discretized_data, est = p.apply(hack_data)

        bn = ContinuousBN()
        info = p.info

        bn.add_nodes(info) # here you will get an error

    learn_bn()
    # The error:
    # 2023-12-14 16:20:05,010 | ERROR    | base.py-add_nodes-0090 | Continuous BN does not support discrete data

Remove output:

.. code-block:: python

    bamt_logger.switch_console_out(False)
    learn_bn() # only KeyError from Python

After this you will no longer receive messages from all loggers of BAMT.

To revert changes just use:

.. code-block:: python

    bamt_logger.switch_console_out(True)
    learn_bn()

    # return
    # 2023-12-14 16:20:05,010 | ERROR    | base.py-add_nodes-0090 | Continuous BN does not support discrete data

How to turn on/off log files for BAMT?
______________________________________

In order to redirect errors to log file:

.. code-block:: python

    bamt_logger.switch_file_out(True,
                                log_file="<absolute/path/to/my_log.log>") # only absolute path
    learn_bn()
    # log file
    # 2023-12-14 16:34:23,414 | ERROR    | base.py-add_nodes-0090 | Continuous BN does not support discrete data


To revert this (it will not delete log created before):

.. code-block:: python

    bamt_logger.switch_file_out(False) # only absolute path
    learn_bn()
    # log file: no new messages.