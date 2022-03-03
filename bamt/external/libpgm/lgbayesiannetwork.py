# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the CyberPoint International, LLC nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
This module provides tools to represent and handle Bayesian networks with linear Gaussian conditional probability distributions.

A linear Gaussian distribution means that the node has a continuous range of outcomes, with a normal distribution over those outcomes. This normal distribution can be parameterized by a *mean* and a *variance*. A linear Gaussian means that the variance of the node is fixed, while the mean is a linear function of the outcomes of the node's parents. In math terms, the mean :math:`m(u)` of a node *u* is a linear function of the values :math:`x_1,\\dots,x_n` of its parents, each weighted by some coefficient :math:`\\beta_i`:

.. math::

    m(u) = \\beta_0 +\\beta_1 x_1 + \\dots + \\beta_n x_n

Linear Gaussians are simple but widely used in statistical modeling.

'''

import random
import math
import sys

from bamt.external.libpgm.orderedskeleton import OrderedSkeleton


class LGBayesianNetwork(OrderedSkeleton):
    '''
    This class represents a Bayesian network with linear Gaussian CPDs. It contains the attributes *V*, *E*, and *Vdata*, as well as the method *randomsample*. 
   
    '''

    def __init__(self, orderedskeleton=None, nodedata=None):
        '''
        This class can be called either with or without arguments. If it is called without arguments, none of its attributes are instantiated and it is left to the user to instantiate them manually. If it is called with arguments, the attributes will be loaded directly from the inputs. The arguments must be (in order):

            1. *orderedskeleton* -- An instance of the :doc:`OrderedSkeleton <orderedskeleton>` or :doc:`GraphSkeleton <graphskeleton>` (as long as it's ordered) class.
            2. *nodedata* -- An instance of the :doc:`NodeData <nodedata>` class.

        If these arguments are present, all attributes of the class (*V*, *E*, and *Vdata*) will be automatically copied from the graph skeleton and node data inputs.
        
        This class requires that the *Vdata* attribute gets loaded with a dictionary with node data of the following fomat::

            "vertex": {
                "parents": ["<name of parent 1>", ... , "<name of parent n>"],
                "children": ["<name of child 1>", ... , "<name of child n>"],   
                "mean_base": <the base mean of the Gaussian distribution>,
                "mean_scal": [<scalar for parent 1 outcome>, ... , <scalar for parent n outcome>],
                "variance": <variance of the Gaussian distibution>
            }

        Note that additional keys are possible in the dict of each vertex.

        Upon loading, the class will also check that the keys of *Vdata* correspond to the vertices in *V*.

        '''
        if (orderedskeleton != None and nodedata != None):
            try:
                self.V = orderedskeleton.V
                '''A list of the names of the vertices.'''
                self.E = orderedskeleton.E
                '''A list of [origin, destination] pairs of vertices that make edges.'''
                self.Vdata = nodedata.Vdata
                '''A dictionary containing CPD data for the nodes.'''
            except: 
                raise (Exception, "Inputs were malformed; first arg must contain V and E attributes and second arg must contain Vdata attribute.")

            assert sorted(self.V) == sorted(self.Vdata.keys()), "Vertices did not match node data"

    def randomsample(self, n, evidence=None, mode="normal"):
        '''
        Produce *n* random samples from the Bayesian Network and return them in a list. 
       
        See above for how the means of linear Gaussians are calculated during sampling.

        This function takes the following arguments:

            1. *n* -- The number of random samples to produce.
            2. *evidence* -- (Optional) A dict containing (vertex: value) pairs that describe the evidence. To be used carefully because it does manually overrides the nodes with evidence instead of affecting the joint probability distribution of the entire graph.
            3. *mode* -- (Optional) Can be set to "verbose", whereupon the method will return a [value, mean, variance] list for each node rather than just the actual value.  
        
        And returns:
            A list of *n* independent random samples, each element of which is a dict containing (vertex: value) pairs.

        Usage example: this would generate a sequence of 10 random samples::
            
            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.lgbayesiannetwork import LGBayesianNetwork

            # load nodedata and graphskeleton
            nd = NodeData()
            skel = GraphSkeleton()
            nd.load("../tests/unittestlgdict.txt")  # an input file
            skel.load("../tests/unittestdict.txt")

            # topologically order graphskeleton
            skel.toporder()

            # load bayesian network
            lgbn = LGBayesianNetwork(skel, nd)

            # sample 
            result = lgbn.randomsample(10)

            # output
            print json.dumps(result, indent=2)

        '''
        assert (isinstance(n, int) and n > 0), "Argument must be a positive integer."

        random.seed()
        seq = []
        distribseq = []
        for _ in range(n):
            outcome = dict()
            distribs = dict()
            for vertex in self.V:
                outcome[vertex] = "default"

            def assignnode(s):
            
                if (evidence != None):
                    if s in evidence.keys():
                        return [evidence[s], ["given value", "given value"]]

                # calculate Bayesian parameters (mean and variance)
                mean = self.Vdata[s]["mean_base"]
                if (self.Vdata[s]["parents"] != None):
                    for x in range(len(self.Vdata[s]["parents"])):
                        parent = self.Vdata[s]["parents"][x]
                        assert outcome[parent] != 'default', "Graph skeleton was not topologically ordered."
                        mean += outcome[parent] * self.Vdata[s]["mean_scal"][x]
                variance = self.Vdata[s]["variance"]

                distribution = [mean, variance]

                # draw random outcome from Gaussian 
                return [random.gauss(mean, math.sqrt(variance)), distribution]          

            for s in self.V:
                if (outcome[s] == "default"):
                    pair = assignnode(s)
                    outcome[s] = pair[0]
                    distribs[s] = pair[1]
            seq.append(outcome)
            distribseq.append(distribs)

        if mode == "normal":
            return seq

        elif mode == "verbose":
            result = []
            for x in range(len(seq)):
                result.append(dict())
                for node in seq[x]: 
                     result[x][node] = [seq[x][node], distribseq[x][node][0], distribseq[x][node][1]] 
            return result
