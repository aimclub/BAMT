"""
**************************
Discretize Continuous Data
**************************

Since pyBN only handles Discrete Bayesian Networks,
and therefore only handles discrete data, it is
important to have effective functions for 
discretizing continuous data. This code aims to
meet that goal.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
from copy import copy
import random
import pandas as pd



"""
>>dediscretize
Function to convert labels back to continuous values

>>data_with_empty_column
Helper function to fill with gaps in a desired column in data


"""









def discretize(data, cols=None, bins=None):
	"""
	Discretize the passed-in dataset. These
	functions will rely on numpy and scipy
	for speed and accuracy... no need to
	reinvent the wheel here. Therefore, pyBN's
	discretization methods are basically just
	wrappers for existing methods.

	The bin number defaults to FIVE (5) for
	all columns if not passed in.

	Arguments
	---------
	*data* : a nested numpy array

	*cols* : a list of integers (optional)
		Which columns to discretize .. defaults
		to ALL columns

	*bins* : a list of integers (optional)
		The number of bins into which each column
		array will be split .. defaults to 5 for
		all columns

	Returns
	-------
	*data* : a discretized copy of original data

    *dic_list* : A dictionary with the value of the middle of each interval
    key = number of the interval
    value = the middle of the interval

    *steps*:  A list with step value of each column in *cols*

	Effects
	-------
	None

	Notes
	-----
	- Should probably add more methods of discretization
		based on mean/median/mode, etc.
	"""
	if bins is not None:
		assert (isinstance(bins, list)), 'bins argument must be a list'
	else:
		try:
			bins = [5]*data.shape[1]
		except ValueError:
			bins = [5]
	
	if cols is not None:
		assert (isinstance(cols,list)), 'cols argument must be a list'
	else:	
		try:
			cols = range(data.shape[1])
		except ValueError:
			cols = [0]

	data = copy(data)
	dic_list = []
	steps = []

	minmax = list(zip(np.amin(data,axis=0),np.amax(data,axis=0)))
	for i, c in enumerate(cols):
		# get min and max of each column
		_min, _max = minmax[c]
		# create the bins from np.linspace
		_bins, step = np.linspace(_min,_max,bins[i], retstep=True)
		# discretize with np.digitize(col,bins)
		data[:,c] = np.digitize(data[:,c],_bins)
		steps.append(int(step))
		dic = {}
		for j in range(1, bins[i]+1, 1):
			dic[j] = (_bins[j-1]) + int(step/2)
		dic_list.append(dic)
	return np.array(data,dtype=np.int32,copy=False), dic_list, steps



def dediscretize(data, dic_labels, steps, cols):
	
	data = copy(data)
	for i, c in enumerate(cols):
		for j in range(data.shape[0]):
			data[j, c] = dic_labels[i][data[j, c]] + random.randint(0,int((steps[i])/2)) - random.randint(0,int((steps[i])/2))
	return pd.DataFrame(data)

def data_with_empty_column(data, index_list):
    empty_data = []
    for e in data.values:
        empty_data.append(list(e.astype('str')))
    for i in range(len(empty_data)):
        for j in index_list:
            empty_data[i][j] = None
    return empty_data









