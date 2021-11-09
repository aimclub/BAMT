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
This module provides tools to represent and handle dynamic Bayesian networks with discrete conditional probability distributions. Dynamic Bayesian networks represent systems that change over time. This means that each node in the BN has a finite number of outcomes, the distribution over which is dependent on the outcomes of the node's parents and on the outcomes of the Bayesian network at the previous time interval. In other words, the Bayesian network changes over time according to Bayesian conditional probability rules.

'''

import random
import sys
from external.libpgm.orderedskeleton import OrderedSkeleton

class DynDiscBayesianNetwork(OrderedSkeleton):
    '''
    This class represents a dynamic Bayesian network with discrete CPD tables. It contains the attributes *V*, *E*, *initial_Vdata*, and *twotbn_Vdata*, and the method *randomsample*.
    
    '''

    def __init__(self, orderedskeleton=None, nodedata=None):
        '''
        This class can be called either with or without arguments. If it is called without arguments, none of its attributes are instantiated and it is left to the user to instantiate them manually. If it is called with arguments, the attributes will be loaded directly from the inputs. The arguments must be (in order):

            1. *orderedskeleton* -- An instance of the :doc:`OrderedSkeleton <orderedskeleton>` or :doc:`GraphSkeleton <graphskeleton>` (as long as it's ordered) class.
            2. *nodedata* -- An instance of the :doc:`NodeData <nodedata>` class.
        
        If these arguments are present, all attributes of the class (*V*, *E*, and *Vdata*) will be automatically copied from the graph skeleton and node data inputs.

        This class requires that the *initial_Vdata* and *twotbn_Vdata* attributes get loaded with a dictionary with node data of the following fomat::
        
            {
                "initial_Vdata": {
                    "<vertex 1>": <dict containing vertex 1 data>,
                    ...
                    "<vertex n>": <dict containing vertex n data>
                }
                "twotbn_Vdata": {
                    "<vertex 1>": <dict containing vertex 1 data>,
                    ...
                    "<vertex n>": <dict containing vertex n data>
                }
            }

        In particular, the ``"parents"`` attribute of ``"twotbn_Vdata"`` has the following format::
            
            "twotbn_Vdata": {
                "vertex": {
                    "parents": ["past_<vertex 1>",...,"past_<vertex n>", "vertex 1",..., "vertex m"]
                    ...
                }
            }

        Where vertices 1 through *n* come from the previous time interval, and vertices 1 through *m* come from the current time interval. Note that additional keys besides the ones listed are possible in the dict of each vertex. For a full example see :doc:`unittestdyndict`.

        Upon loading, the class will also check that the keys of *Vdata* correspond to the vertices in *V*.

        '''
        if (orderedskeleton != None and nodedata != None):
            try:
                self.V = orderedskeleton.V
                '''A list of the names of the vertices.'''
                self.E = orderedskeleton.E
                '''A list of [origin, destination] pairs of vertices that make edges.'''
                self.initial_Vdata = nodedata.initial_Vdata
                '''A dictionary containing CPD data for the Bayesian network at time interval 0.'''
                self.twotbn_Vdata = nodedata.twotbn_Vdata
                '''A dictionary containing CPD data for the Bayesian network for time intervals greater than 0.'''
            except:
                raise (Exception, "Inputs were malformed; first arg must contain V and E attributes and second arg must contain initial_Vdata and twotbn_Vdata attributes.")

            # check that inputs match up
            assert (sorted(self.V) == sorted(self.initial_Vdata.keys())), ("initial_Vdata vertices did not match vertex data:", self.V, self.Vdata.keys())
            assert (sorted(self.V) == sorted(self.twotbn_Vdata.keys())), ("twotbn_Vdata vertices did not match vertex data:", self.V, self.Vdata.keys())
    
    def randomsample(self, n):
        '''
        This method produces a sequence of length *n* containing one dynamic Bayesian network sample over *n* time units. In other words, the first entry of the sequence is a sample from the initial Bayesian network, and each subsequent entry is sampled from the Bayesian network conditioned on the outcomes of its predecessor. This function requires a specific dictionary format in Vdata, as shown in :doc:`dynamic discrete bayesian network<unittestdyndict>`.
            
        This function takes the following arguments:
            1. *n* -- The number of time units over which to sample (thus also the length of the sequence produced)
        
        And returns:
            A list of *n* random samples, each conditioned on its precedessor, each a dict containing (vertex: value) pairs.
        
        Usage example: this would produce a sequence of 10 samples, one per time step, each conditioned on its predecessor::

            import json
            
            from graphskeleton import GraphSkeleton
            from dyndiscbayesiannetwork import DynDiscBayesianNetwork

            path = "../tests/unittestdyndict.txt" # an input file
            f = open(path, 'r')
            g = eval(f.read())

            d = DynDiscBayesianNetwork()
            skel = GraphSkeleton()
            skel.V = g["V"]
            skel.E = g["E"]
            skel.toporder()
            d.V = skel.V
            d.E = skel.E
            d.initial_Vdata = g["initial_Vdata"]
            d.twotbn_Vdata = g["twotbn_Vdata"]

            seq = d.randomsample(10)
            print json.dumps(seq, indent=2)
        
        '''
        assert (isinstance(n, int) and n > 0), "Argument must be a positive integer."

        random.seed()
        seq = []
        for t in range(n):
            outcome = dict()
            for vertex in self.V:
                outcome[vertex] = "default"
            
            def assignnode(s):
                
                # find entry in dictionary and store
                if t == 0: 
                    Vdataentry = self.initial_Vdata[s]
                else:
                    Vdataentry = self.twotbn_Vdata[s]

                p = Vdataentry["parents"]
                if (not p):
                    distribution = Vdataentry["cprob"]
                else:

                    # find parents from previous time step (if necessary)
                    pvalues = []
                    for parent in p:
                        if parent[:5] == "past_":
                            pvalues.append(str(seq[t-1][parent[5:]]))
                        else:
                            pvalues.append(str(outcome[parent]))
                    for pvalue in pvalues:
                        assert pvalue != 'default', "Graph skeleton was not topologically ordered."
                       
                    distribution = Vdataentry["cprob"][str(pvalues)]

                # choose interval
                rand = random.random()
                lbound = 0 
                ubound = 0
                for interval in range(int(Vdataentry["numoutcomes"])):
                    ubound += distribution[interval]
                    if (lbound <= rand and rand < ubound):
                        rindex = interval
                        break
                    else:
                        lbound = ubound 
            
                return Vdataentry["vals"][rindex]
            
            for s in self.V:
                if (outcome[s] == "default"):
                    outcome[s] = assignnode(s)
            
            seq.append(outcome)
        return seq
     

