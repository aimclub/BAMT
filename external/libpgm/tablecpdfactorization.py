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
This module provides tools for creating and using factorized representations of Bayesian networks. Factorized representations of Bayesian networks are discrete CPDs whose values have been flattened into a single array, while the cardinalities and strides of each variable represented are kept track of separately. With the proper setup, these flattened structures can be more easily multiplied together, reduced, and operated on. For more information on factors cf. Koller et al. Ch. 4.

'''

from external.libpgm.tablecpdfactor import TableCPDFactor

import random
import copy

class TableCPDFactorization():
    '''
    This class represents a factorized Bayesian network with discrete CPD tables. It contains the attributes *bn*, *originalfactorlist*, and *factorlist*, and the methods *refresh*, *sumproductve*, *sumproducteliminatevar*, *condprobve*, *specificquery*, and *gibbssample*.

    '''

    def __init__(self, bn):
        '''
        This class is constructed with a :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` instance as argument. First, it takes the input itself and stores it in the *bn* attribute. Then, it transforms the information of each of these nodes from standard discrete CPD form into a :doc:`TableCPDFactor <tablecpdfactor>` isntance and stores the instances in an array in the attribute *originalfactorlist*. Finally, it makes a copy of this list to work with and stores it in *factorlist*.
        
        '''
        assert hasattr(bn, "V") and hasattr(bn, "E") and hasattr(bn, "Vdata"), \
            "Input must be a DiscreteBayesianNetwork instance."
        
        self.bn = bn
        '''The Bayesian network used as argument at instantiation.'''
        self.originalfactorlist = []
        '''A list of :doc:`TableCPDFactor <tablecpdfactor>` instances, one per node.'''
        for vertex in bn.V:
            factor = TableCPDFactor(vertex, bn)
            self.originalfactorlist.append(factor)
        self.factorlist = copy.deepcopy(self.originalfactorlist)  
        '''A working copy of *originalfactorlist*.'''

        assert self.factorlist, "Factor list not properly loaded, check for an incomplete class instance as input."
    
    def refresh(self):
        '''
        Refresh the *factorlist* attribute to equate with *originalfactorlist*. This is in effect a reset of the system, erasing any changes to *factorlist* that the program has executed.

        '''
        self.factorlist = copy.deepcopy(self.originalfactorlist)    
            
    def sumproducteliminatevar(self, vertex):    
        '''
        Multiply the all the factors in *factorlist* that have *vertex* in their scope, then sum out *vertex* from the resulting product factor. Replace all factors that were multiplied together with the resulting summed-out product.
        
        Arguments:
            1. *vertex* - The name of the variable to eliminate.
        
        Attributes modified:
            1. *factorlist* -- Modified to reflect the eliminated variable.
        
        For more information on this algorithm cf. Koller et al. 298

        '''
        factors2 = []
        factors1 = []
        for factor in self.factorlist:
            try:
                factor.scope.index(vertex)
                factors1.append(factor)
            except ValueError:
                factors2.append(factor)
                
        # multiply factors1 array together
        for i in range(1, len(factors1)):
            factors1[0].multiplyfactor(factors1[i])
        
        # sum out the vertex from the factor
        factors1[0].sumout(vertex)
        
        # add to rest of factors and return
        if (factors1[0] != None):
            factors2.append(factors1[0])
        self.factorlist = factors2
    
    def sumproductve(self, vertices):
        '''
        Eliminate each vertex in *vertices* from *factorlist* using *sumproducteliminatevar*.
        
        Arguments:
            1. *vertices* -- A list of UUIDs of vertices to be eliminated.
        
        Attributes modified: 
            1. *factorlist* -- modified to become a single factor representing the remaining variables.

        '''
    
        # eliminate one by one
        for vertex in vertices:
            self.sumproducteliminatevar(vertex)
        
        # multiply together if many factors remain 
        for i in range(1, len(self.factorlist)):
            self.factorlist[0].multiplyfactor(self.factorlist[i])
        
        self.factorlist = self.factorlist[0]
        
    def condprobve(self, query, evidence):
        '''
        Eliminate all variables in *factorlist* except for the ones queried. Adjust all distributions for the evidence given. Return the probability distribution over a set of variables given by the keys of *query* given *evidence*. 
        
        Arguments:
            1. *query* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what outcome to calculate the probability of. 
            2. *evidence* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what is known about the system.
                    
        Attributes modified:
            1. *factorlist* -- Modified to be one factor representing the probability distribution of the query variables given the evidence.
                           
        The function returns *factorlist* after it has been modified as above.
        
        Usage example: this code would return the distribution over a queried node, given evidence::

            import json

            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.nodedata import NodeData
            from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            from external.libpgm.tablecpdfactorization import TableCPDFactorization

            # load nodedata and graphskeleton
            nd = NodeData()
            skel = GraphSkeleton()
            nd.load("../tests/unittestdict.txt")
            skel.load("../tests/unittestdict.txt")

            # toporder graph skeleton
            skel.toporder()

            # load evidence
            evidence = dict(Letter='weak')
            query = dict(Grade='A')

            # load bayesian network
            bn = DiscreteBayesianNetwork(skel, nd)

            # load factorization
            fn = TableCPDFactorization(bn)

            # calculate probability distribution
            result = fn.condprobve(query, evidence)

            # output
            print json.dumps(result.vals, indent=2)
            print json.dumps(result.scope, indent=2)
            print json.dumps(result.card, indent=2)
            print json.dumps(result.stride, indent=2)

        '''
        assert (isinstance(query, dict) and isinstance(evidence, dict)), "First and second args must be dicts."

        eliminate = self.bn.V[:]
        for key in query.keys():
            eliminate.remove(key)
        for key in evidence.keys():
            eliminate.remove(key)
        
        # modify factors to account for E = e
        for key in evidence.keys():
            for x in range(len(self.factorlist)):
                if (self.factorlist[x].scope.count(key) > 0):
                    self.factorlist[x].reducefactor(key, evidence[key])
            for x in reversed(range(len(self.factorlist))):
                if (self.factorlist[x].scope == []):
                    del(self.factorlist[x])
                    
        # eliminate all necessary variables in the new factor set to produce result
        self.sumproductve(eliminate)
        
        # normalize result
        summ = 0
        lngth = len(self.factorlist.vals)
        for x in range(lngth):
            summ += self.factorlist.vals[x]
        for x in range(lngth):
            self.factorlist.vals[x] /= summ
            
        # return table
        return self.factorlist

    def specificquery(self, query, evidence):
        '''
        Eliminate all variables except for the ones specified by *query*. Adjust all distributions to reflect *evidence*. Return the entry that matches the exact probability of a specific event, as specified by *query*.
        
        Arguments:
            1. *query* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what outcome to calculate the probability of. The value must be a list of values (for ordinary queries do a list of length one).
            2. *evidence* -- A dict containing (key: value) pairs reflecting (variable: value) evidence that is known about the system.
                    
        Attributes modified:
            1. *factorlist* -- Modified as in *condprobve*.
                           
        The function then chooses the entries of *factorlist* that match the queried event or events. It then operates on them to return the probability that the event (or events) specified will occur, represented as a float between 0 and 1.

        Note that in this function, queries of the type P((x=A or x=B) and (y=C or y=D)) are permitted. They are executed by formatting the *query* dictionary like so::

            {
                "x": ["A", "B"],
                "y": ["C", "D"]
            }
        
        Usage example: this code would answer the specific query that vertex ``Grade`` gets outcome ``A`` given that ``Letter`` has outcome ``weak``, in :doc:`this Bayesian network <unittestdict>`::

            import json

            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.nodedata import NodeData
            from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            from external.libpgm.tablecpdfactorization import TableCPDFactorization
            
            # load nodedata and graphskeleton
            nd = NodeData()
            skel = GraphSkeleton()
            nd.load("../tests/unittestdict.txt")
            skel.load("../tests/unittestdict.txt")

            # toporder graph skeleton
            skel.toporder()

            # load evidence
            evidence = dict(Letter='weak')
            query = dict(Grade=['A'])

            # load bayesian network
            bn = DiscreteBayesianNetwork(skel, nd)

            # load factorization
            fn = TableCPDFactorization(bn)

            # calculate probability distribution
            result = fn.specificquery(query, evidence)

            # output
            print result

        '''
        assert (isinstance(query, dict) and isinstance(evidence, dict)), "First and second args must be dicts."
        assert query, "Query must be non-empty."
        
        self.condprobve(query, evidence)

        # now self.factorlist contains the joint distribution across the
        # variables designated in query. next, we have to locate the entries
        # where the variables have values matching the query (e.g., where "Grade"
        # is "A" and "Intelligence" is "High"). because must loop once for each 
        # variable, and we don't know how many variables there are, we use 
        # recursion to iterate through the variables
        visited = dict()
        rindices = dict()
        findices = []
        
        # find corresponding numbers to possible values, store in rindices
        for var in query.keys():
            rindices[var] = []
            visited[var] = False
            for poss in query[var]:
                rindices[var].append(self.bn.Vdata[var]["vals"].index(poss))
        
        # define function to help iterate recursively through all combinations of variables
        def findentry(var, index):
            visited[var] = True 
        
            for x in range(len(rindices[var])):
                newindex = index + rindices[var][x] * self.factorlist.stride[var]
                if (visited.values().count(False) > 0):
                    i = visited.values().index(False)
                    nextvar = visited.keys()[i]
                    findentry(nextvar, newindex)
                else:
                    # we've accounted for all variable assignments and found an entry
                    findices.append(newindex)
            visited[var] = False
            return
        
        # calculate all relevant entries
        findentry(visited.keys()[0], 0)
            
        # sum entries
        fanswer = 0
        for findex in findices:
            fanswer += self.factorlist.vals[findex]
            
        # return result
        return fanswer

    def gibbssample(self, evidence, n):
        '''
        Return a sequence of *n* samples using the Gibbs sampling method, given evidence specified by *evidence*. Gibbs sampling is a technique wherein for each sample, each variable in turn is erased and calculated conditioned on the outcomes of its neighbors. This method starts by sampling from the 'prior distribution,' which is the distribution not conditioned on evidence, but the samples provably get closer and closer to the posterior distribution, which is the distribution conditioned on the evidence. It is thus a good way to deal with evidence when generating random samples.
        
        Arguments: 
            1. *evidence* -- A dict containing (key: value) pairs reflecting (variable: value) that represents what is known about the system.
            2. *n* -- The number of samples to return.
        
        Returns:
        
        A list of *n* random samples, each element of which is a dict containing (vertex: value) pairs.
        
        For more information, cf. Koller et al. Ch. 12.3.1

        Usage example: This code would generate a sequence of 10 samples::

            import json
            
            from external.libpgm.graphskeleton import GraphSkeleton
            from external.libpgm.nodedata import NodeData
            from external.libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
            from external.libpgm.tablecpdfactorization import TableCPDFactorization
            
            # load nodedata and graphskeleton
            nd = NodeData()
            skel = GraphSkeleton()
            nd.load("../tests/unittestdict.txt")
            skel.load("../tests/unittestdict.txt")

            # toporder graph skeleton
            skel.toporder()

            # load evidence
            evidence = dict(Letter='weak')

            # load bayesian network
            bn = DiscreteBayesianNetwork(skel, nd)

            # load factorization
            fn = TableCPDFactorization(bn)

            # sample 
            result = fn.gibbssample(evidence, 10)

            # output
            print json.dumps(result, indent=2)

        '''
        self.refresh()
        random.seed() 
        
        # declare result array
        seq = []
        
        # create initial instantiation 
        initial = self.bn.randomsample(1)
        for key in evidence.keys():
            initial[0][key] = evidence[key]
        seq.append(initial[0])  
        
        # find nodes that we are sampling
        order = []
        for vertex in self.bn.V:
            if vertex not in evidence.keys():
                order.append(vertex)

        # reduce factorlist given E = e
        for key in evidence.keys():
            for x in range(len(self.factorlist)):
                if (self.factorlist[x].scope.count(key) > 0):
                    self.factorlist[x].reducefactor(key, evidence[key])
            for x in reversed(range(len(self.factorlist))):    
                if (self.factorlist[x].scope == []):
                    del(self.factorlist[x])
                    
        # define function to create the next instantiation
        def next(current): 
            for node in order:
                # multiply all relevant factors together
                relevantfactors = []
                for factor in self.factorlist:
                    if (factor.scope.count(node) > 0):
                        factorcopy = factor.copy()
                        relevantfactors.append(factorcopy)
                for j in range(1, len(relevantfactors)):
                    relevantfactors[0].multiplyfactor(relevantfactors[j])

                # reduce to leave only the one node
                for othernode in order:
                    if (othernode != node and relevantfactors[0].scope.count(othernode) > 0):
                        relevantfactors[0].reducefactor(othernode, current[othernode])
                
                # renormalize
                summ = 0
                for val in relevantfactors[0].vals:
                    summ += val
                for x in range(len(relevantfactors[0].vals)):
                    relevantfactors[0].vals[x] /= summ
                
                # convert random number
                val = random.random()
                lboundary = 0
                uboundary = 0
                for x in range(len(relevantfactors[0].vals)):
                    uboundary += relevantfactors[0].vals[x]
                    if (lboundary <= val and val < uboundary):
                        rindex = x
                        # print s, val
                        break
                    else:
                        lboundary = uboundary 
                
                # modify result
                current[node] = self.bn.Vdata[node]["vals"][rindex]
   
            return current
                        
        # run next() function n times
        for u in range(n-1):
            copy = dict() 
            for entry in seq[u]:
                copy[entry] = seq[u][entry]
            seq.append(next(copy))
            
        # return all samples
        return seq
