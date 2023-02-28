Bayesian Networks Learning Algorithms
=====================================

A Bayesian network is a graphical probabilistic model that is a directed acyclic graph in which nodes are features in the data and edges are conditional dependencies between features.

.. image::
    .. image:: ../../images/bnla_model.png
    :target:: ../../images/bnla_model.png
    :align: center

The use of Bayesian networks implies the existence of algorithms for learning the structure and parameters of distributions at the nodes.

Structure Learning of Bayesian Networks
---------------------------------------

Now, the library implements structural learning algorithms that consider the task of learning the BN as an optimization problem:

.. math::
    V_{opt}, E_{opt}=\underset{G' \subset G_{possible}}{argmax}F(G')

The following scoring functions are implemented as evaluation functions of the network quality: *K2*, *BIC*, *AIC* and *MI*. 
Moreover, the *BIC*, *AIC* and *MI* metrics can be used both on discretized data and on mixed ones 
(for more details, see the `publication <https://www.sciencedirect.com/science/article/pii/S1877050921020925>`__).  

As an optimization algorithm, the greedy Hill Climbing algorithm is used, which iteratively changes the structure and remembers the change that leads to the greatest increase in score:

.. image::
    .. image:: ../../images/HC_scheme_disser.png
    :target:: ../../images/HC_scheme_disser.png
    :align: center

Since greedy algorithms have their drawbacks, we plan to add evolutionary algorithms for structure learning in the future. 
Also, our framework allows you to include expert knowledge in the process of structural network learning.
This is done by limiting the search space (white edges and black edges),
it is also possible to set the starting graph from which optimization begins and to restrict or allow the removal of the edges of the starting graph.
All this allows you to flexibly use expert knowledge in the learning process. 

Also in this framework, a variant of structural learning is proposed, when connections from continuous to discrete nodes are allowed (has_logit=true).
The fact is that for a long time Bayesian networks were unable to model conditional distributions in a discrete node with continuous parents.
The solution was then found using classification models, however if you want to limit the appearance of such relationships this can be done with the ‘has_logit’ flag. 


Parameter Learning of Bayesian Networks
---------------------------------------

In addition to structural learning in Bayesian networks, there is learning of distribution parameters at nodes. As a parameter learning algorithm, this framework implements the likelihood maximization algorithm. With the available dataset 
*D*, it is necessary to select an estimate for θ parameter that satisfies the condition: 

.. math::
    L(\widehat{\theta}:D)=\underset{\theta\subset\Theta }{max}L(\theta:D)

By default, distributions at network nodes are modeled using Gaussian distributions and linear regression for continuous nodes and conditional probability tables (CPT) for discrete nodes.

.. image::
    .. image:: ../../images/params_learning.png
    :target:: ../../images/params_learning.png
    :align: center

However, this approach does not model real data very well, in which there is a clear non-Gaussianity and non-linearity.
For such cases, it is proposed, for example, to use mixtures of Gaussian distributions, since with a sufficiently large number of components, the mixture can describe a distribution of any shape (parameter use_mixture).
In the framework for parametric learning, automatic selection of the number of components is implemented.

.. image::
    .. image:: ../../images/mixture_edge.png
    :target:: ../../images/mixture_edge.png
    :align: center

A non-parametric approach to the representation of distributions is also implemented, when any machine learning model is used to predict the parameters of the conditional distribution,
so the conditional mathematical expectation can be predicted quite accurately, and the conditional dispersion is a prediction error.
You can choose this method with the help of the ``set_classifier()`` and ``set_regressor()`` methods.
Now the user must choose the model himself, but in the future an algorithm for automatic selection of models will be added. 

.. image::
    .. image:: ../../images/logit_net.png
    :target:: ../../images/logit_net.png
    :align: center
