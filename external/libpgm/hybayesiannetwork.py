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
This module provides tools to represent and handle Bayesian networks with conditional probability distributions that can be specified node-by-node. 

This method allows for the construction of a Bayesian network with every combination of every type of CPD, provided that the user provides a method for sampling each type of node and stores this method in the proper place, namely as the ``choose()`` method of a class in ``libpgm.CPDtypes/``.

'''

import random

from external.libpgm.orderedskeleton import OrderedSkeleton


class HyBayesianNetwork(OrderedSkeleton):
    '''
    This class represents a Bayesian network with CPDs of any type. The nodes of the Bayesian network are stored first in a dictionary, specifying their "type", which should be descriptive ('discrete', 'lg', etc.). Furthermore, the types of each node associate them with a class found in ``libpgm/CPDtypes/``. The nodes are then stored also as instances of classes found in this directory. The purpose of this is that each node has its own method for being sampled given the outcomes of its parents.
    
    '''

    def __init__(self, orderedskeleton=None, nodedata=None):
        '''
        This class can be called either with or without arguments. If it is called without arguments, none of its attributes are instantiated and it is left to the user to instantiate them manually. If it is called with arguments, the attributes will be loaded directly from the inputs. The arguments must be (in order):

            1. *orderedskeleton* -- An instance of the :doc:`OrderedSkeleton <orderedskeleton>` or :doc:`GraphSkeleton <graphskeleton>` (as long as it's ordered) class.
            2. *nodedata* -- An instance of the :doc:`NodeData <nodedata>` class.
        
        It is required that the *nodedata* class instance inputted has its *nodes* attribute instantiated. In order for this to be the case, the instance must have run its *entriestoinstances* method.

        If the arguments above are present, all attributes of the class (*V*, *E*, *Vdata*, and *nodes*) will be automatically copied from the graph skeleton and node data inputs.

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

                # specific to hybrid Bayesian network
                self.nodes = nodedata.nodes
                '''A dictionary of {key: value} pairs linking the node name (the key) to a class instance (the value) representing the node, its node data, and its sampling function.'''
            except:
                raise (Exception,
                       "Inputs were malformed; first arg must contain V and E attributes and second arg must contain Vdata and nodes attributes.")

            # check that inputs match up
            assert sorted(self.V) == sorted(self.Vdata.keys()), "Node data did not match graph skeleton nodes."

    def randomsample(self, n, method, evidence=None):
        '''
        Produce *n* random samples from the Bayesian networki, subject to *evidence*, and return them in a list. This function requires the *nodes* attribute to be instantiated.
        
        This function takes the following arguments:

            1. *n* -- The number of random samples to produce.
            2. *evidence* -- (Optional) A dict containing (vertex: value) pairs that describe the evidence. To be used carefully because it does manually overrides the nodes with evidence instead of affecting the joint probability distribution of the entire graph.
        
        And returns:
            A list of *n* independent random samples, each element of which is a dict containing (vertex: value) pairs.
        
        Usage example: this would generate a sequence of 10 random samples::
            
            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.hybayesiannetwork import HyBayesianNetwork

            # load nodedata and graphskeleton
            nd = NodeData()
            skel = GraphSkeleton()
            nd.load("../tests/unittesthdict.txt")   # an input file
            skel.load("../tests/unittestdict.txt")

            # topologically order graphskeleton
            skel.toporder()

            # convert nodes to class instances
            nd.entriestoinstances()

            # load bayesian network
            hybn = HyBayesianNetwork(skel, nd)

            # sample 
            result = hybn.randomsample(10)

            # output
            print json.dumps(result, indent=2)
            


        '''
        assert ((isinstance(n, int) and n > 0), "Argument must be a positive integer.")

        random.seed()
        seq = []
        for _ in range(n):
            outcome = dict()
            for vertex in self.V:
                outcome[vertex] = "default"

            def assignnode(name, node):

                # check if node is already observed
                if (evidence != None):
                    if name in evidence.keys():
                        return evidence[name]

                # get parent values
                p = self.getparents(name)
                if (p == []):
                    pvalues = []
                else:
                    pvalues = [outcome[t] for t in self.Vdata[name]["parents"]]  # ideally can we pull this from the skeleton so as not to store parent data at all?
                    for pvalue in pvalues:
                        assert (pvalue != 'default', "Graph skeleton was not topologically ordered.")

                # use built in function to determine outcome
                
                return node.choose(pvalues, method, outcome)

            for s in self.V:
                if (outcome[s] == "default"):
                    outcome[s] = assignnode(s, self.nodes[s])

            seq.append(outcome)
        return seq
