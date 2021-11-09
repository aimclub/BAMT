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
This module contains tools for representing "LG + D" (linear Gaussian and discrete) nodes -- those with a finite number of outcomes and one or more Gaussian parents -- as class instances with their own *choose* method to choose an outcome for themselves based on parent outcomes.

'''

import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings


class Logit():
    '''
    This class represents a discrete node, as described above. It contains the *Vdataentry* attribute and the *choose* method.

    '''
    def __init__(self, Vdataentry):
        '''
        This class is constructed with the argument *Vdataentry* which must be a dict containing a dictionary entry for this particular node. The dict must contain entries of the following form::
            "classes": <array of str as class names in multinomial logit regression>
            "mean_base": <array of floatы as intercept of logit regression>,
            "mean_scal": <a flat array formed from an array of arrays of floats, 
            to which the respectively ordered continuous parent results for each class are multiplied>,

        The *Vdataentry* attribute is set equal to this *Vdataentry* input upon instantiation.

        '''
        self.Vdataentry = Vdataentry
  
    def choose(self, pvalues, method, outcome):
        '''
        Randomly choose state of node from probability distribution conditioned on *pvalues*.
        This method has two parts: (1) determining the proper probability
        distribution, and (2) using that probability distribution to determine
        an outcome.
        Arguments:
            1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry['parents']``.
        The function creates a Gaussian distribution in the manner described in :doc:`lgbayesiannetwork`, and samples from that distribution, returning its outcome.
        
        '''
        warnings.filterwarnings("ignore", category=FutureWarning)
        pvalues = [str(outcome[t]) for t in self.Vdataentry["parents"]] # ideally can we pull this from the skeleton so as not to store parent data at all?
        for pvalue in pvalues:
            assert pvalue != 'default', "Graph skeleton was not topologically ordered."
        
            
        model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
        
        model.classes_ = np.array(self.Vdataentry["classes"], dtype=str)
        if len(self.Vdataentry["classes"]) > 1:
            model.coef_ = np.array(self.Vdataentry["mean_scal"], dtype=float).reshape(-1,len(self.Vdataentry["parents"]))
            model.intercept_ = np.array(self.Vdataentry["mean_base"], dtype=float)
            distribution = model.predict_proba(np.array(pvalues).reshape(1, -1))[0]
                    
            # choose
            rand = random.random()
            lbound = 0 
            ubound = 0
            for interval in range(len(self.Vdataentry["classes"])):
                ubound += distribution[interval]
                if (lbound <= rand and rand < ubound):
                    rindex = interval
                    break
                else:
                    lbound = ubound 
        
            return str(self.Vdataentry["classes"][rindex])
        
        else:
            return str(self.Vdataentry["classes"][0])