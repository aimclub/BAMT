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
This module contains tools for representing "crazy" nodes -- nodes where the method for sampling is to multiply the crazyinput by -10 or 10 and add :math:`\pi` -- as class instances with their own *choose* method to choose an outcome for themselves based on parent outcomes. 

The existence of this 'crazy' type is meant to indicate the true universality of
the universal sampling method found in :doc:`hybayesiannetwork`. While no CPD would
actually be this crazy, the libary has the setup to support any type of CPD.


'''
import math
import random

class Crazy():
    '''
    This class represents a crazy node, as described above. It contains the *Vdataentry* attribute and the *choose* method.

    '''
    def __init__(self, Vdataentry):
        '''
        This class is constructed with the argument *Vdataentry* which must be a dict containing a dictionary entry for this particualr node. The dict must contain an entry of the following form::

            "crazyinput": <number that is the crazy input>

        This ``"crazyinput"`` entry contains the number that will be used in the crazy sampling function.  The *Vdataentry* attribute is set equal to this *Vdataentry* input upon instantiation.
        '''
        self.Vdataentry = Vdataentry
        '''A dict containing CPD data for the node.'''
        
    def choose(self, pvalues):
        '''
        Randomly choose state of node from probability distribution conditioned on *pvalues*.

        This method has two parts: (1) determining the proper probability
        distribution, and (2) using that probability distribution to determine
        an outcome.

        Arguments:
            1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in self.Vdataentry['parents'].

        The function takes the crazyinput, multiplies it by either 10 or -10 randomly, adds :math:`\\pi`, converts it to a string, and appends the word "bluberries!". It returns this value.

        '''
        crazyinput = self.Vdataentry["crazyinput"]
        answer = "%.2f blueberries!" % (random.choice([10, -10]) * crazyinput + math.pi)
        return answer
