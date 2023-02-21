How to contribute
=================

We highly encourage you to contribute to the project. You can do this by forking the project on GitHub and sending us a pull request. We will review your code and merge it into the project if it is good.

Step-by-step guide
------------------

If you are new to GitHub, here is a step-by-step guide on how to contribute to the project.

#. First, fork `the BAMT project on GitHub <https://github.com/ITMO-NSS-team/BAMT>`__. To do this, click the "Fork" button on the top right of the page. This will create a copy of the project in your own GitHub account.

#. Clone the repository to your local machine by using `GitHub Desktop <https://desktop.github.com/>`__ or the CLI commnad (make sure that you have git installed):

   .. code-block:: bash

      git clone git@github.com:YourUsername/BAMT.git
      cd path/to/repos/BAMT

#. Create a new branch for your changes, it is not recommended to work on the ``master`` branch:

   .. code-block:: bash

      git checkout -b my-new-feature

#. Make sure that your environment is up to date and set up for development. You can install all the dependencies by running the following command inside the project directory:

   .. code-block:: bash

      pip install -r requirements.txt

#. Start making changes on your newly created branch, remembering to never work on the ``master`` branch! Work on this copy on your computer using Git to do the version control.

#. When you're done making changes, check that your changes pass the tests by running the following command inside the project directory or follow `the instructions <https://github.com/ITMO-NSS-team/BAMT/blob/master/tests/README.md>`__. Note, that you need to have the ``pytest`` package installed:

   .. code-block:: bash
    
        pip install pytest
        pytest -v -s tests

#. When you are done editing and testing, commit your changes to your local repository with a descriptive message:

   .. code-block:: bash
      
      git add modified_files
      git commit -am "Added some feature"

#. Push your local changes to the remote repository on GitHub into your branch:

   .. code-block:: bash

      git push origin my-new-feature

Finally, go to the web page of your fork of the FEDOT repo, and click 'Pull Request' (PR) to send your changes to the maintainers for review.

If the following instructions look confusing, check `git documentation <https://git-scm.com/doc>`__ or use GitHub Desktop with GUI. 
Using GitHUb extension for Visual Studio Code, PyCharm or whatever IDE you use is also a good option.

Before submitting a pull request
--------------------------------

Before you submit a pull request for your contribution, please work through this checklist to make sure that you have done everything necessary so we can efficiently review and accept your changes.

If your contribution changes BAMT code in any way, please follow the check list below:

 - Update the `documentation <https://github.com/ITMO-NSS-team/BAMT/blob/master/docs>`__ to reflect the changes.

 - Update the `tests <https://github.com/ITMO-NSS-team/BAMT/blob/master/tests>`__ 

 - Update the `README.rst <https://github.com/ITMO-NSS-team/BAMT/blob/master/README.rst>`__ file if the change affects description of BAMT.

 - Make sure that your code is properly formatted according to the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ standard. You can automatically format your code using the ``autopep8`` package.

 - Make sure your commits are atomic (one feature per commit) and that you have written a descriptive commit message.

 If your contribution requires a new dependency, please make sure that you have added it to the ``requirements.txt`` file and follow these additional steps:

 -  Double-check that the new dependency is easy to install via ``pip`` or ``conda`` and supports Python 3. If the dependency requires a complicated installation, then we most likely won't merge your changes because we want to keep BAMT easy to install.

-  Add the required version of the library to `requirements.txt <https://github.com/ITMO-NSS-team/BAMT/blob/master/requirements.txt>`__


Contribute to the documentation
-------------------------------
Take care of the documentation.

All the documentation is created with the Sphinx autodoc feature. Use ..
automodule:: <module_name> section which describes all the code in the module.

-  If a new package with several scripts:

   #. Go to `docs/source/api <https://github.com/ITMO-NSS-team/BAMT/blob/master/docs/source/api>`__ and create new your_name_for_file.rst file.

   #. Add a Header underlined with “=” sign. It’s crucial.

   #. Add automodule description for each of your scripts

      .. code-block::

         $.. automodule:: bamt.your.first.script.path
         $   :members:
         $   :undoc-members:
         $   :show-inheritance:

         $.. automodule:: bamt.your.second.script.path
         $   :members:
         $   :undoc-members:
         $   :show-inheritance:

   #. Add your_name_for_file to the toctree at docs/index.rst

-  If a new module to the existed package:

    Most of the sections are already described in `docs/source/api <https://github.com/nccr-itmo/FEDOT/tree/master/docs/source/api>`__ , so you can:

   -  choose the most appropriate and repeat 3-d step from the previous section.
   -  or create a new one and repeat 2-3 steps from the previous section.

-  If a new function or a class to the existing module:

    Be happy. Everything is already done for you.


Acknowledgements
----------------

This guide document is based at well-written `TPOT Framework contribution guide <https://github.com/EpistasisLab/tpot/blob/master/docs_sources/contributing.md>`__ and `FEDOT Framework contribution guide <https://raw.githubusercontent.com/aimclub/FEDOT/master/docs/source/contribution.rst>`__.