from pgmpy.estimators.StructureScore import StructureScore
from bamt.utils.MathUtils import *
from sklearn.mixture import GaussianMixture
from gmr import GMM
import numpy as np
import sys
import multiprocessing
from multiprocessing import Pool
from time import time



    

def BIC_local(data):
    v = []
    par = []
    if data.shape[1] == 1:
        v.append(data.columns[0])
    else:
        v.append(data.columns[0])
        par = data.columns[1:]
    
    n_comp = int((component(data, data.columns, 'aic') + component(data, data.columns, 'bic')) / 2)
    #gmm_model = GaussianMixture(n_components=n_comp).fit(data.values)
    #gmm_model = GMM(n_components=n_comp, priors=gmm_model.weights_, means=gmm_model.means_, covariances=gmm_model.covariances_)
    gmm_model = GMM(n_components=n_comp).from_samples(data.values)
    index = [j for j in range(1, len(par) + 1)]
    ll = 0

    if data.shape[1] != 1:
        def process_chunk(row):
            # df_chunk.reset_index(inplace=True, drop=True)
            #for i in (df_chunk.shape[0]):
            # par_val = df_chunk.loc[i, par].values
            # v_val = df_chunk.loc[i, v].values
            cond_log = np.log(gmm_model.condition(index, row[1:].values).to_probability_density(row[0])[0])
            return cond_log
        res = data.apply(process_chunk, axis=1)
        ll = np.sum(res)


    else:

        def process_chunk_simple(row):
            # df_chunk.reset_index(inplace=True, drop=True)
            #for i in (df_chunk.shape[0]):
            # par_val = df_chunk.loc[i, par].values
            # v_val = df_chunk.loc[i, v].values
            cond_log = np.log(gmm_model.to_probability_density(row[0])[0])
            return cond_log
        res = data.apply(process_chunk_simple, axis=1)
        ll = np.sum(res)
    ll = ll - (np.log(data.shape[0]) / 2) * data.shape[1]
    
    return round(ll)




def BIC_local_gauss(data):
    v = []
    par = []
    if data.shape[1] == 1:
        v.append(data.columns[0])
    else:
        v.append(data.columns[0])
        par = data.columns[1:]
    ll = 0
    n_comp = 1
    #gmm_model = GaussianMixture(n_components=n_comp).fit(data.values)
    #gmm_model = GMM(n_components=n_comp, priors=gmm_model.weights_, means=gmm_model.means_, covariances=gmm_model.covariances_)
    gmm_model = GMM(n_components=n_comp).from_samples(data.values)

    if data.shape[1] != 1:
        
        index = [j for j in range(1, len(par) + 1)]
        for i in range(data.shape[0]):
            par_val = data.loc[i, par].values
            v_val = data.loc[i, v].values
            cond_log = np.log(gmm_model.condition(index, par_val).to_probability_density(v_val)[0])
            ll += cond_log
    else:
        for i in range(data.shape[0]):
            ll_log = np.log(gmm_model.to_probability_density(data.loc[i,v].values)[0]) 
            ll += ll_log
    ll = ll - (np.log(data.shape[0]) / 2) * data.shape[1]
    
    return round(ll)



class BICGMM(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for log-likelihood for BayesianModels with GMM.
        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        """
        super(BICGMM, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""
        nrow = len(self.data)
        list_var = [variable]
        list_var.extend(parents)       
        score = BIC_local(self.data[list_var])

        return score


class BICGauss(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for log-likelihood for BayesianModels with GMM.
        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        """
        super(BICGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""
        nrow = len(self.data)
        list_var = [variable]
        list_var.extend(parents)       
        score = BIC_local_gauss(self.data[list_var])

        return score




