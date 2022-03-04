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
This module contains tools for representing "LG + D" (linear Gaussian and discrete) nodes -- those with a finite number of outcomes, one or more Gaussian parents, and one or more discrete parents -- as class instances with their own *choose* method to choose an outcome for themselves based on parent outcomes.

'''

import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

class Logitandd():
    '''
    This class represents a discrete node, as described above. It contains the *Vdataentry* attribute and the *choose* method.

    '''
    def __init__(self, Vdataentry):
        '''
		This class is constructed with the argument *Vdataentry* which must be a dict containing a dictionary entry for this particualr node. The dict must contain an entry of the following form::

			"cprob": {
				"['<parent 1, value 1>',...,'<parent n, value 1>']": {
								"classes": <array of str as class names in multinomial logit regression>
                                "mean_base": <array of floatы as intercept of logit regression>,
                                "mean_scal": <a flat array formed from an array of arrays of floats, 
                                to which the respectively ordered continuous parent results for each class are multiplied>,
							}
				...
				"['<parent 1, value j>',...,'<parent n, value k>']": {
								"classes": <array of str as class names in multinomial logit regression>
                                "mean_base": <array of floatы as intercept of logit regression>,
                                "mean_scal": <a flat array formed from an array of arrays of floats, 
                                to which the respectively ordered continuous parent results for each class are multiplied>,
							}
			}

		This ``"cprob"`` entry contains parameters for multinomial logit regression for each combination of discrete parents.  The *Vdataentry* attribute is set equal to this *Vdataentry* input upon instantiation.

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
        rand = random.random()

        dispvals = []
        lgpvals = []
        for pval in pvalues:
            if (isinstance(pval, str)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
        # find correct Gaussian
        lgdistribution = self.Vdataentry["hybcprob"][str(dispvals)]
       
        for pvalue in lgpvals:
            assert pvalue != 'default', "Graph skeleton was not topologically ordered."
        
            
        model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
        
        model.classes_ = np.array(lgdistribution["classes"], dtype=object)
       
        if len(lgdistribution["classes"]) > 1:
            model.coef_ = np.array(lgdistribution["mean_scal"], dtype=float).reshape(-1,len(lgpvals))
            model.intercept_ = np.array(lgdistribution["mean_base"], dtype=float)
            distribution = model.predict_proba(np.array(lgpvals).reshape(1, -1))[0]
                    
            # choose
            rand = random.random()
            lbound = 0 
            ubound = 0
            for interval in range(len(lgdistribution["classes"])):
                ubound += distribution[interval]
                if (lbound <= rand and rand < ubound):
                    rindex = interval
                    break
                else:
                    lbound = ubound 
        
            return str(lgdistribution["classes"][rindex])
        
        else:
            return str(lgdistribution["classes"][0])
