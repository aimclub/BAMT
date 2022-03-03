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
This module provides tools for creating and using graph skeletons for Bayesian networks. A graph skeleton in this case is a vertex set and a directed edge set, with no further information about the specific nodes. 

'''
from bamt.external.libpgm.dictionary import Dictionary

import sys

class GraphSkeleton(Dictionary):
    '''
    This class represents a graph skeleton, meaning a vertex set and a directed edge set. It contains the attributes *V* and *E*, and the methods *load*, *getparents*, *getchildren*, and *toporder*.
    
    '''

    def __init__(self):
        self.V = None
        '''A list of names of vertices.'''
        self.E = None
        '''A list of [origin, destination] pairs of vertices that constitute edges.'''
        self.alldata = None
        '''(Inherited from dictionary) A variable that stores a key-indexable dictionary once it is loaded from a file.'''

    def load(self, path):
        '''
        Load the graph skeleton from a text file located at *path*. 
        
        Text file must be a plaintext .txt file with a JSON-style representation of a dict.  Dict must contain the top-level keys "V" and "E" with the following formats::

            {
                'V': ['<vertex_name_1>', ... , '<vertex_name_n'],
                'E': [['vertex_of_origin', 'vertex_of_destination'], ... ]
            }
        
        Arguments:
            1. *path* -- The path to the file containing input data (e.g., "mydictionary.txt").
        
        Attributes modified: 
            1. *V* -- The set of vertices. 
            2. *E* -- The set of edges.

        '''
        self.dictload(path)
        self.V = self.alldata["V"]
        self.E = self.alldata["E"]

        # free unused memory
        del self.alldata
        
    def getparents(self, vertex):
        '''
        Return the parents of *vertex* in the graph skeleton.
        
        Arguments:
            1. *vertex* -- The name of the vertex whose parents the function finds.
        
        Returns:
            A list containing the names of the parents of the vertex.

        '''
        assert (vertex in self.V), "The graph skeleton does not contain this vertex."

        parents = []
        for pair in self.E:
            if (pair[1] == vertex):
                parents.append(pair[0])
        return parents
    
    def getchildren(self, vertex):
        '''
        Return the children of *vertex* in the graph skeleton. 
        
        Arguments:
            1. *vertex* -- The name of the vertex whose children the function finds.
        
        Returns:
            A list containing the names of the children of the vertex.

        '''
        assert (vertex in self.V), "The graph skeleton does not contain this vertex."

        children = []
        for pair in self.E:
            if (pair[0] == vertex):
                children.append(pair[1])
        return children
    
    def toporder(self):
        '''
        Modify the vertices of the graph skeleton such that they are in topological order. 

        A topological order is an order of vertices such that if there is an edge from *u* to *v*, *u* appears before *v* in the ordering. It works only for directed ayclic graphs.
        
        Attributes modified:
            1. *V* -- The names of the vertices are put in topological order.
        
        The function also checks for cycles in the graph, and returns an error if one is found.

        '''
        Ecopy = [x[:] for x in self.E]
        roots = [] 
        toporder = []
      
        for vertex in self.V:
            # find roots
            if (self.getparents(vertex) == []):
                roots.append(vertex)
       
        while roots != []:
            n = roots.pop()
            toporder.append(n)
            for edge in reversed(Ecopy):
                if edge[0] == n:
                    m = edge[1]
                    Ecopy.remove(edge)
                    yesparent = False 
                    for e in Ecopy:
                        if e[1] == m:
                            yesparent = True
                            break
                    if yesparent == False:
                        roots.append(m)
        assert (not Ecopy), ("Graph contains a cycle", Ecopy)
        self.V = toporder 

