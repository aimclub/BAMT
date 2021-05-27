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
This module provides tools for creating and using an individual factorized representation of a node. See description of factorized representations in :doc:`tablecpdfactorization`.

'''

import sys 

class TableCPDFactor(object):
    '''
    This class represents a factorized representation of a conditional probability distribution table. It contains the attributes *inputvertex*, *inputbn*, *vals*, *scope*, *stride*, and *card*, and the methods *multiplyfactor*, *sumout*, *reducefactor*, and *copy*. 

    '''

    def __init__(self, vertex, bn):
        '''
        This class is constructed with a :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` instance and a *vertex* name as arguments. First it stores these inputs in *inputvertex* and *inputbn*. Then, it creates a factorized representation of *vertex*, storing the values in *vals*, the names of the variables involved in *scope* the cardinality of each of these variables in *card* and the stride of each of these variables in *stride*.
        
        '''
        self.inputvertex = vertex
        '''The name of the vertex.'''
        self.inputbn = bn
        '''The :doc:`DiscreteBayesianNetwork <discretebayesiannetwork>` instance that the vertex lives in.'''
        
        result = dict( vals = [], stride = dict(), card = [], scope = [])
        root = bn.Vdata[vertex]["cprob"]
        
        # add values
        def explore(_dict, key, depth, totaldepth):
            if depth == totaldepth:
                for x in _dict[str(key)]:
                    result["vals"].append(x)
                return
            else:
                for val in bn.Vdata[bn.Vdata[vertex]["parents"][depth]]["vals"]:
                    ckey = key[:]
                    ckey.append(str(val))
                    explore(_dict, ckey, depth+1, totaldepth)
                    
        if not bn.Vdata[vertex]["parents"]:
            result["vals"] = bn.Vdata[vertex]["cprob"]
        else: 
            td = len(bn.Vdata[vertex]["parents"])
            explore(root, [], 0, td)
        
        # add cardinalities
        result["card"].append(bn.Vdata[vertex]["numoutcomes"])
        if (bn.Vdata[vertex]["parents"] != None):
            for parent in reversed(bn.Vdata[vertex]["parents"]):
                result["card"].append(bn.Vdata[parent]["numoutcomes"])
            
        # add scope
        result["scope"].append(vertex)
        if (bn.Vdata[vertex]["parents"] != None):
            for parent in reversed(bn.Vdata[vertex]["parents"]):
                result["scope"].append(parent)
        
        
        # add strides
        stride = 1
        result["stride"] = dict()
        for x in range(len(result["scope"])):
            result["stride"][result["scope"][x]] = (stride)
            stride *= bn.Vdata[result["scope"][x]]["numoutcomes"]
        
        self.vals = result["vals"]
        '''A flat array of all the values from the CPD.'''
        self.scope = result["scope"]
        '''An array of vertices that affect the vals found in *vals*. Normally, this is the node itself and its parents.'''
        self.card = result["card"]
        '''A list of the cardinalities of each vertex in scope, where cardinality is the number of values that the vertex may take. The cardinalities are indexed according to the vertex's index in *scope*.'''
        self.stride = result["stride"]
        '''A dict of {vertex: value} pairs for each vertex in *scope*, where vertex is the name of the vertex and value is the stride of that vertex in the *vals* array.'''

    def multiplyfactor(self, other):  # cf. PGM 359 
        '''
        Multiply the factor by another :doc:`TableCPDFactor <tablecpdfactor>`. Multiplying factors means taking the union of the scopes, and for each combination of variables in the scope, multiplying together the probabilities from each factor that that combination will be found.
        
        Arguments:
            1. *other* -- An instance of the :doc:`TableCPDFactor <tablecpdfactor>` class representing the factor to multiply by.
                 
        Attributes modified: 
            *vals*, *scope*, *stride*, *card* -- Modified to reflect the data of the new product factor.
                                                         
        For more information cf. Koller et al. 359.

        '''
        if (not isinstance(other, TableCPDFactor)):
            msg = "Error: in method 'multiplyfactor', input was not a TableCPDFactor instance"
            sys.exit(msg)
        j = 0
        k = 0
        result = dict()
        
        # merge scopes
        result["scope"] = self.scope
        result["card"] = self.card
        for x in range(len(other.scope)):
            try:
                result["scope"].index(other.scope[x])
            except: 
                result["scope"].append(other.scope[x])
                result["card"].append(other.card[x])
    
        # calculate possible combinations of scope variables
        possiblevals = 1
        for val in result["card"]:
            possiblevals *= val
        
        # algorithm (see book)
        assignment = [0 for l in range(len(result["scope"]))]
        result["vals"] = []
        for _ in range(possiblevals):
            result["vals"].append(self.vals[j] * other.vals[k])
            for l in range(len(result["scope"])):
                assignment[l] = assignment[l] + 1
                if (assignment[l] == result["card"][l]):
                    assignment[l] = 0
                    try:
                        j = j - (result["card"][l] - 1) * self.stride[result["scope"][l]]
                    except:
                        pass
                    try:
                        k = k - (result["card"][l] - 1) * other.stride[result["scope"][l]]
                    except: 
                        pass
                else:
                    try: 
                        j = j + self.stride[result["scope"][l]]
                    except: 
                        pass
                    try: 
                        k = k + other.stride[result["scope"][l]]
                    except:
                        pass
                    break
            
        # add strides
        stride = 1 
        result["stride"] = dict()
        for x in range(len(result["scope"])):
            result["stride"][result["scope"][x]] = (stride)
            stride *= result["card"][x]
    
        self.vals = result["vals"]
        self.scope = result["scope"]
        self.card = result["card"]
        self.stride = result["stride"]
        
    def sumout(self, vertex):
        '''
        Sum out the variable specified by *vertex* from the factor. Summing out means summing all sets of entries together where *vertex* is the only variable changing in the set. Then *vertex* is removed from the scope of the factor.
        
        Arguments:
            1. *vertex* -- The name of the variable to be summed out.
        
        Attributes modified: 
            *vals*, *scope*, *stride*, *card* -- Modified to reflect the data of the summed-out product factor.
        
        For more information see Koller et al. 297.

        '''
        if (self.scope.count(vertex) == 0):
            msg = "Error: in method 'sumout', vertex '%s' not in scope of factor" % (vertex)
            sys.exit(msg)
        vscope = self.scope.index(vertex)
        vstride = self.stride[vertex]
        vcard = self.card[vscope]
        result = [0 for i in range(len(self.vals)/self.card[vscope])]
        
        # machinery that calculates values in summed out factor
        k = 0
        lcardproduct = 1
        for i in range(vscope):
            lcardproduct *= self.card[i]
        for i in range(len(result)):
            for h in range(vcard):
                result[i] += self.vals[k + (vstride * h)]
            k += 1
            if (k % lcardproduct == 0):
                k += (lcardproduct * (vcard - 1))
        self.vals = result
        
        # modify scope, card, and stride in new factor
        self.scope.remove(vertex)
        del(self.card[vscope])
        for i in range(vscope, len(self.stride)-1):
            self.stride[self.scope[i]] /= vcard
        del(self.stride[vertex])
        
    def reducefactor(self, vertex, value):
        '''
        Reduce the factor knowing that *vertex* equals *value*. Reducing the factor means erasing all possibilities for *vertex* other than *value* from the distribution, and removing *vertex* from the scope.
        
        Arguments:
            1. *vertex* -- The UUID of the variable whose outcome is known.
            2. *value* -- The known outcome of that variable.
        
        Attributes modified: 
            *vals*, *scope*, *stride*, *card* -- Modified to reflect the data of the reduced factor.

        '''
        vscope = self.scope.index(vertex)
        vstride = self.stride[vertex]
        vcard = self.card[vscope]
        result = [0 for i in range(len(self.vals)/self.card[vscope])]
        
        # added step: find value index from evidence
        try:
            index = self.inputbn.Vdata[vertex]['vals'].index(value)
        except:
            raise (Exception, "Second arg was not a possible value of first arg.")
     
        # machinery that calculates values in summed out factor
        k = 0
        lcardproduct = 1
        for i in range(vscope):
            lcardproduct *= self.card[i]
        for i in range(len(result)):
            result[i] += self.vals[k + (vstride * index)]
            k += 1
            if (k % lcardproduct == 0):
                k += (lcardproduct * (vcard - 1))
        self.vals = result
        
        # modify scope, card, and stride in new factor
        self.scope.remove(vertex)
        del(self.card[vscope])
        for i in range(vscope, len(self.stride)-1):
            self.stride[self.scope[i]] /= vcard
        del(self.stride[vertex])
    
    def copy(self):
        '''Return a copy of the factor.'''
        copy = TableCPDFactor(self.inputvertex, self.inputbn)
        copy.vals = self.vals[:]
        copy.stride = self.stride.copy()
        copy.scope = self.scope[:]
        copy.card = self.card[:]
        return copy
    
        
    
