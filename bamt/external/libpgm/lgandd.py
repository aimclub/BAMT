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
This module contains tools for representing "LG + D" (linear Gaussian and discrete) nodes -- those with a Gaussian distribution, one or more Gaussian parents, and one or more discrete parents -- as class instances with their own *choose* method to choose an outcome for themselves based on parent outcomes.

'''
import random
import math
from scipy.spatial import distance
import numpy as np
from pomegranate import DiscreteDistribution
import time
from gmr import GMM


class Lgandd():
	'''
	This class represents a LG + D node, as described above. It contains the *Vdataentry* attribute and the *choose* method

	'''
	def __init__(self, Vdataentry):
		'''
		This class is constructed with the argument *Vdataentry* which must be a dict containing a dictionary entry for this particualr node. The dict must contain an entry of the following form::

			"cprob": {
				"['<parent 1, value 1>',...,'<parent n, value 1>']": {
								"mean_base": <float used for mean starting point
											  (\mu_0)>,
								"mean_scal": <array of scalars by which to
											  multiply respectively ordered 
											  continuous parent outcomes>,
								"variance": <float for variance>
							}
				...
				"['<parent 1, value j>',...,'<parent n, value k>']": {
								"mean_base": <float used for mean starting point
											  (\mu_0)>,
								"mean_scal": <array of scalars by which to
											  multiply respectively ordered 
											  continuous parent outcomes>,
								"variance": <float for variance>
							}
			}

		This ``"cprob"`` entry contains a linear Gaussian distribution (conditioned on the Gaussian parents) for each combination of discrete parents.  The *Vdataentry* attribute is set equal to this *Vdataentry* input upon instantiation.

		'''
		self.Vdataentry = Vdataentry
		'''A dict containing CPD data for the node.'''

	def choose(self, pvalues, method, outcome):
		'''
		Randomly choose state of node from probability distribution conditioned on *pvalues*.
		This method has two parts: (1) determining the proper probability
		distribution, and (2) using that probability distribution to determine
		an outcome.
		Arguments:
			1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry['parents']``.
		The function goes to the entry of ``"cprob"`` that matches the outcomes of its discrete parents. Then, it constructs a Gaussian distribution based on its Gaussian parents and the parameters found at that entry. Last, it samples from that distribution and returns its outcome.
		'''
		random.seed()
		sample = 0

		if method == 'simple':
			dispvals = []
			lgpvals = []
			for pval in pvalues:
				if ((isinstance(pval, str)) | ((isinstance(pval, int)))):
					dispvals.append(pval)
				else:
					lgpvals.append(pval)
			lgdistribution = self.Vdataentry["hybcprob"][str(dispvals)]
			mean = lgdistribution["mean_base"]
			if (self.Vdataentry["parents"] != None):
				for x in range(len(lgpvals)):
					if (lgpvals[x] != "default"):
						mean += lgpvals[x] * lgdistribution["mean_scal"][x]
					else:
						print ("Attempted to sample node with unassigned parents.")
			variance = lgdistribution["variance"]
			sample = random.gauss(mean, math.sqrt(variance))  
		else:
			dispvals = []
			lgpvals = []
			for pval in pvalues:
				if ((isinstance(pval, str)) | ((isinstance(pval, int)))):
					dispvals.append(pval)
				else:
					lgpvals.append(pval)
			lgdistribution = self.Vdataentry["hybcprob"][str(dispvals)]
			mean = lgdistribution["mean_base"]
			variance = lgdistribution["variance"]
			w = lgdistribution["mean_scal"]
			if len(w) != 0:
				if (len(lgpvals) != 0):
					indexes = [i for i in range (1, (len(lgpvals)+1), 1)]
					if not np.isnan(np.array(lgpvals)).all():
						n_comp = len(w)
						gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
						sample = gmm.predict(indexes, [lgpvals])[0][0]
					else:
						sample = np.nan
				else:
					n_comp = len(w)
					gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
					sample = gmm.sample(1)[0][0]
			else:
				sample = np.nan
			

		return sample    


	def choose_gmm(self, pvalues, outcome):
		'''
		Randomly choose state of node from probability distribution conditioned on *pvalues*.
		This method has two parts: (1) determining the proper probability
		distribution, and (2) using that probability distribution to determine
		an outcome.
		Arguments:
			1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry['parents']``.
		The function creates a Gaussian distribution in the manner described in :doc:`lgbayesiannetwork`, and samples from that distribution, returning its outcome.
		
		'''
		random.seed()

		# split parents by type
		dispvals = []
		lgpvals = []
		for pval in pvalues:
			if (isinstance(pval, str)):
				dispvals.append(pval)
			else:
				lgpvals.append(pval)
	  

		# error check
		try: 
			a = dispvals[0]
			a = lgpvals[0]
		except IndexError:
			#print ("Did not find LG and discrete type parents.")
			s = "Did not find LG and discrete type parents."

		# find correct Gaussian
		lgdistribution = self.Vdataentry["hybcprob"][str(dispvals)]
		s = 0

		# calculate Bayesian parameters (mean and variance)
		mean = lgdistribution["mean_base"]
		variance = lgdistribution["variance"]
		w = lgdistribution["mean_scal"]
		#if (self.Vdataentry["parents"] != None):
		if (len(lgpvals) != 0):
			if isinstance(mean, list):
				indexes = [i for i in range (1, (len(lgpvals)+1), 1)]
				# for x in range(len(lgpvals)):
				# 	if (lgpvals[x] != "default"):
				# 		X.append(lgpvals[x])
				# 	else:
				# 	# temporary error check 
				# 		print ("Attempted to sample node with unassigned parents.")
				if not np.isnan(np.array(lgpvals)).any():
					n_comp = len(w)
					gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
					s = gmm.predict(indexes, [lgpvals])[0][0]
				else:
					s = np.nan
			else:
				for x in range(len(lgpvals)):
					if (lgpvals[x] != "default"):
						mean += lgpvals[x] * w[x]
					else:
					# temporary error check 
						print ("Attempted to sample node with unassigned parents.")	
				s = random.gauss(mean, math.sqrt(variance))
		else:
			if isinstance(mean, list):
				n_comp = len(w)
				gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
				s = gmm.sample(1)
				s = s[0][0]
			else:
				s = random.gauss(mean, math.sqrt(variance))
		

			
		# draw random outcome from Gaussian
		# note that this built in function takes the standard deviation, not the
		# variance, thus requiring a square root
		return s

	def choose_mix(self, pvalues, outcome):
		'''
		Randomly choose state of node from probability distribution conditioned on *pvalues*.

		This method has two parts: (1) determining the proper probability
		distribution, and (2) using that probability distribution to determine
		an outcome.

		Arguments:
			1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry['parents']``.

		The function goes to the entry of ``"cprob"`` that matches the outcomes of its discrete parents. Then, it constructs a Gaussian distribution based on its Gaussian parents and the parameters found at that entry. Last, it samples from that distribution and returns its outcome.

		'''
		random.seed()

		# split parents by type
		dispvals = []
		lgpvals = []
		for pval in pvalues:
			if (isinstance(pval, str)):
				dispvals.append(pval)
			else:
				lgpvals.append(pval)
	  

		# error check
		try: 
			a = dispvals[0]
			a = lgpvals[0]
		except IndexError:
			#print ("Did not find LG and discrete type parents.")
			s = "Did not find LG and discrete type parents."

		# find correct Gaussian
		lgdistribution = self.Vdataentry["hybcprob"][str(dispvals)]
		# mean_node = 0
		# variance_node = 1

		# calculate Bayesian parameters (mean and variance)
		mean = lgdistribution["mean_base"]
		variance = lgdistribution["variance"]
		if isinstance(mean, list):
			parents = []
			parents_mean = []
			if (len(lgpvals)) != 0:
				if (self.Vdataentry["parents"] != None):
					for x in range(len(lgpvals)):
						if (lgpvals[x] != "default"):
							parents.append(lgpvals[x])
							parents_mean.append(lgdistribution["mean_scal"][x])
						else:
							print ("Attempted to sample node with unassigned parents.")
					if len(parents) == 1:
						if str(parents[0]) != 'nan':
							dists = []
							for vector in parents_mean:
								try:
									dists.append(distance.euclidean(parents, vector))
								except:
									dists.append(100000)
							label = dists.index(min(dists))
							mean_node = mean[label]
							variance_node = variance[label]
						else:
							mean_node = mean[random.randint(0,4)]
							variance_node = variance[random.randint(0,4)]
					else:
						if np.nan in parents:
							if parents.count(np.nan) < len(parents):
								nan_index = [i for i,d in enumerate(parents) if str(d)=='nan']
								dists = []
								for vector in parents_mean:
									try:
										dists.append(distance.euclidean([p for p in parents if parents.index(p) not in nan_index], [p for p in vector if vector.index(p) not in nan_index]))
									except:
										dists.append(100000000)
									label = dists.index(min(dists))
									mean_node = mean[label]
									variance_node = variance[label]
							else:
								mean_node = mean[random.randint(0,4)]
								variance_node = variance[random.randint(0,4)]
						else:
							dists = []
							for vector in parents_mean:
								try:
									dists.append(distance.euclidean(parents, vector))
								except:
									dists.append(100000000)
							label = dists.index(min(dists))
							mean_node = mean[label]
							variance_node = variance[label]

					# dists = []
					# for vector in parents_mean:
						
					#     dists.append(distance.euclidean(parents, vector))
						
					# label = dists.index(min(dists))
					# mean_node = mean[label]
					# variance_node = variance[label]
			else:
				label = random.randint(0,4)
				mean_node = mean[label]
				variance_node = variance[label]
		else:
			if (self.Vdataentry["parents"] != None):
				for x in range(len(lgpvals)):
					if (lgpvals[x] != "default"):
						mean += lgpvals[x] * lgdistribution["mean_scal"][x]
					else:

					# temporary error check 
						print ("Attempted to sample node with unassigned parents.")
			mean_node = mean
			variance_node = variance
			


		
		# if (self.Vdataentry["parents"] != None):
		#     for x in range(len(lgpvals)):
		#         if (lgpvals[x] != "default"):
		#             mean += lgpvals[x] * lgdistribution["mean_scal"][x]
		#         else:

		#             # temporary error check 
		#             print ("Attempted to sample node with unassigned parents.")

		#variance = lgdistribution["variance"]

		# draw random outcome from Gaussian (I love python)
		return random.gauss(mean_node, math.sqrt(variance_node))          
