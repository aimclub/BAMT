.. _install:

Installation
============

The easiest way to get BAMT is through pip using the command

.. code-block:: bash

    pip install bamt

Since BAMT has LightGBM in it's dependencies it is required to install gcc and cmake
to compile LightGBM for Unix-based systems.

On Linux you can do that with any package manager you prefer:


.. code-block:: bash

    sudo apt-get install gcc cmake
    pacman -S gcc cmake


On macOS you can use homebrew:

.. code-block:: bash

    brew install gcc cmake

To learn more about Windows installation of LightGBM or other details,
please follow
`official LightGBM documentation page <https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html>`__.
