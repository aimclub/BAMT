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
This module facilitates the process of creating ordered graph skeletons by topologically ordering them automatically.

'''

from bamt.external.libpgm.graphskeleton import GraphSkeleton


class OrderedSkeleton(GraphSkeleton):
    '''
    This class represents a graph skeleton (see :doc:`graphskeleton`) that is always topologically ordered.
    
    '''
    
    def __init__(self, graphskeleton=None):
        self.V = None
        '''A list of names of vertices'''
        self.E = None
        '''A list of [origin, destination] pairs of verties that constitute edges.'''

    def load(self, path):
        '''Loads a dictionary from a file located at *path* in the same manner as :doc:`graphskeleton`, but includes a step where it topologically orders the nodes.'''
    
        self.dictload(path)
        self.V = self.alldata["V"]
        self.E = self.alldata["E"]

        # topologically order
        self.toporder()

        # free unused memory
        del self.alldata
