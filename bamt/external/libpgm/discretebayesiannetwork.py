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
This module provides tools to represent and handle Bayesian networks with discrete conditional probability distribuitions. This means that each node has a finite number of outcomes, the distribution over which is dependent on the outcomes of the node's parents.

'''

import json
import random

from core import core_utils as bntutils

from bamt.external.libpgm.orderedskeleton import OrderedSkeleton
from bamt.external.libpgm.tablecpdfactorization import TableCPDFactorization


class DiscreteBayesianNetwork(OrderedSkeleton):
    '''
    This class represents a Bayesian network with discrete CPD tables. It contains the attributes *V*, *E*, and *Vdata*, as well as the method *randomsample*.   
    
    '''

    def __init__(self, orderedskeleton=None, nodedata=None, path=None):
        '''
        This class can be called either with or without arguments. If it is called without arguments, none of its attributes are instantiated and it is left to the user to instantiate them manually. If it is called with arguments, the attributes will be loaded directly from the inputs. Note that the user must specify EITHER *nodedata* and *orderedskeleton* OR *path*.

            1. *orderedskeleton* -- An instance of the :doc:`OrderedSkeleton <orderedskeleton>` or :doc:`GraphSkeleton <graphskeleton>` (as long as it's ordered) class.
            2. *nodedata* -- An instance of the :doc:`NodeData <nodedata>` class.
            3. *path* -- The path to a file containing complete, properly formatted json for a discrete Bayesian network. See :doc:`unittestdict` for an example.
        
        If these arguments are present, all attributes of the class (*V*, *E*, and *Vdata*) will be automatically copied from the graph skeleton and node data inputs.

        This class requires that the *Vdata* attribute gets loaded with a dictionary with node data of the following fomat::


        Note that additional keys besides the ones listed are possible in the dict of each vertex. For a full example see :doc:`unittestdict`.

        Upon loading, the class will also check that the keys of *Vdata* correspond to the vertices in *V*.
        '''
        assert not (orderedskeleton and nodedata and path), "specify nodedata and orderedskeleton OR path"
        if (orderedskeleton != None and nodedata != None):
            try:
                self.V = orderedskeleton.V
                '''A list of the names of the vertices.'''
                self.E = orderedskeleton.E
                '''A list of [origin, destination] pairs of vertices that make edges.'''
                self.Vdata = nodedata.Vdata
                '''
                A dictionary containing CPD data for the nodes of the format:: 

                    "vertex": {
                        "numoutcomes": <number of possible outcome values>,
                        "vals": ["<name of value 1>", ... , "<name of value n>"],
                        "parents": ["<name of parent 1>", ... , "<name of parent n>"],
                        "children": ["<name of child 1>", ... , "<name of child n>"],   
                        "cprob": {
                            "['<parent 1, value 1>',...,'<parent n, value 1>']": [<probability of vals[0]>, ... , <probability of vals[n-1]>],
                            ...
                            "['<parent 1, value j>',...,'<parent n, value k>']": [<probability of vals[0]>, ... , <probability of vals[n-1]>],
                        }
                    }
    
                '''
            except:
                raise (Exception,
                       "Inputs were malformed; first arg must contain V and E attributes and second arg must contain Vdata attribute.")

            # check that inputs match up
            assert (sorted(self.V) == sorted(self.Vdata.keys())), (
            "Vertices did not match vertex data:", self.V, self.Vdata.keys())

        if (path):

            # validate
            with open(path) as f:
                try:
                    j = json.load(f)
                except:
                    raise bntextError("json not properly formatted: failed to load")
            bntutils.refresh(path)
            bntutils._validate(path)

            # load
            self.V = j["V"]
            self.E = j["E"]
            self.Vdata = j["Vdata"]

    def specificquery(self, query, evidence):
        '''
        .. note: Shortcut method to the *specificquery* method in :doc:`tablecpdfactorization`

        Eliminate all variables except for the ones specified by *query*. Adjust all distributions to reflect *evidence*. Return the entry that matches the exact probability of a specific event, as specified by *query*.
        
        Arguments:
            1. *query* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what outcome to calculate the probability of. The value of the query is a list of one or more values that can be taken by the variable.
            2. *evidence* -- A dict containing (key: value) pairs reflecting (variable: value) evidence that is known about the system. 
                    
        Returns:
            - the probability that the event (or events) specified will occur, represented as a float between 0 and 1.

        Note that in this function, queries of the type P((x=A or x=B) and (y=C or y=D)) are permitted. They are executed by formatting the *query* dictionary like so::

            {
                "x": ["A", "B"],
                "y": ["C", "D"]
            }

        '''
        # validate
        if not (hasattr(self, "V") and hasattr(self, "E") and hasattr(self, "Vdata")):
            raise notloadedError("Bayesian network is missing essential attributes")
        assert isinstance(query, dict) and isinstance(evidence, dict), "query and evidence must be dicts"
        for k in query.keys():
            assert isinstance(query[k], list), "the values of your query must be lists, even if singletons"

            # calculate
        fn = TableCPDFactorization(self)
        return fn.specificquery(query, evidence)

    def randomsample(self, n, evidence=None):
        '''
        Produce *n* random samples from the Bayesian network, subject to *evidence*, and return them in a list.             

        This function takes the following arguments:

            1. *n* -- The number of random samples to produce.
            2. *evidence* -- (Optional) A dict containing (vertex: value) pairs that describe the evidence. To be used carefully because it does manually overrides the nodes with evidence instead of affecting the joint probability distribution of the entire graph.
        
        And returns:
            A list of *n* independent random samples, each element of which is a dict containing (vertex: value) pairs.
        
        Usage example: this would generate a sequence of 10 random samples::
            
            import json

            from external.libpgm.nodedata import NodeData
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            
            # load nodedata and graphskeleton
            nd = NodeData()
            skel = GraphSkeleton()
            nd.load("../tests/unittestdict.txt")    # any input file
            skel.load("../tests/unittestdict.txt")

            # topologically order graphskeleton
            skel.toporder()

            # load bayesian network
            bn = DiscreteBayesianNetwork(skel, nd)

            # sample 
            result = bn.randomsample(10)

            # output
            print json.dumps(result, indent=2)


        '''
        assert (isinstance(n, int) and n > 0), "Argument must be a positive integer."

        random.seed()
        seq = []
        for _ in range(n):
            outcome = dict()
            for vertex in self.V:
                outcome[vertex] = "default"

            def assignnode(s):

                if (evidence != None):
                    if s in evidence.keys():
                        return evidence[s]

                # find entry in dictionary and store
                Vdataentry = self.Vdata[s]

                # slice up [0, 1) into intervals
                p = Vdataentry["parents"]
                if (not p):
                    distribution = Vdataentry["cprob"]
                else:
                    pvalues = [str(outcome[t]) for t in Vdataentry[
                        "parents"]]  # ideally can we pull this from the skeleton so as not to store parent data at all?
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
