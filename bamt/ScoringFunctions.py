from tkinter import N
from pgmpy.estimators.StructureScore import StructureScore
from bamt.utils.MathUtils import *
from sklearn.mixture import GaussianMixture
from gmr import GMM
import numpy as np
import sys
import multiprocessing
from multiprocessing import Pool
from time import time
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from scipy.stats import norm
import math
import numba
from bamt.mi_entropy_gauss import mi_gauss
from bamt.redef_info_scores import log_lik_local, BIC_local, AIC_local

    

def BIC_local_gmm(data, var, parents):
    if len(parents) == 0:
        cols = [var]+parents
        data = data[cols]
        N = data.shape[0]
        d = len(cols)
        #n_comp = int((component(data, cols, 'aic') + component(data, cols, 'bic')) / 2)
        n_comp = number_of_components(data, cols)
        #n_comp = number_of_components_k_means(data, cols)
        gmm_model = GaussianMixture(n_components=n_comp, init_params='k-means++').fit(data.values)
        ll = np.sum(gmm_model.score(data.values)) - 0.005*np.log(N)*n_comp*(d + 1 + d*(d+1)//2)
    else:
        cols = [var]+parents
        data = data[cols]
        N = data.shape[0]
        d = len(cols)
        #n_comp = int((component(data, cols, 'aic') + component(data, cols, 'bic')) / 2)
        n_comp = number_of_components(data, cols)
        #n_comp = number_of_components_k_means(data, cols)
        gmm_model = GaussianMixture(n_components=n_comp, init_params='k-means++').fit(data.values)
        gmm_model = GMM(n_components=n_comp, priors=gmm_model.weights_, means=gmm_model.means_, covariances=gmm_model.covariances_)
        index = [j for j in range(1, len(parents) + 1)]
        cond_m = gmm_model.predict(index, data[parents].values)
        cond_m = cond_m.reshape(N,)
        std = np.abs(data[var].values - cond_m)
        #variance = math.sqrt(mse(data[var].values, cond_m))
        p = norm(loc=cond_m, scale=std).logpdf(data[var].values)
        p = [0 if math.isnan(x) or math.isinf(x) else x for x in p]
        ll = (np.sum(p)/N) - 0.005*np.log(N)*n_comp*(d + 1 + d*(d+1)//2)

    return round(ll)


    # ll = 0
    # if len(parents) != 0:
    #     data['label'] = gmm_model.predict(data.values)
    #     for i in range(n_comp):
    #         sample = data.loc[data['label'] == i]
    #         n = sample.shape[0]
    #         if n != 0:
    #             m = linear_model.LinearRegression().fit(sample[parents].values, sample[var].values)
    #             predict = m.predict(sample[parents].values)
    #             variance = math.sqrt(mse(sample[var].values, predict))
    #             cond_m = [m.intercept_]*n + np.sum(([m.coef_]*n) * (sample[parents].values), axis=1)
    #             # for j in range(len(parents)):
    #             #     cond_m += ([m.coef_[j]]*n)*sample[parents[j]].values
    #             p = norm(loc=cond_m, scale=variance).logpdf(sample[var].values)
    #             p = [-10000000000000000000 if math.isnan(x) else x for x in p]
    #             ll += np.sum(p) / sample.shape[0]
    #     ll /= n_comp
    # else:
    #     ll = np.sum(gmm_model.score(data.values))
    # N = data.shape[0]
    # d = len(cols)
    # ll = ll - 0.5*np.log(N)*n_comp*(d + 1 + d*(d+1)//2)














    # v = []
    # par = []
    # if data.shape[1] == 1:
    #     v.append(data.columns[0])
    # else:
    #     v.append(data.columns[0])
    #     par = data.columns[1:]
    # ll = 0
    # cols = [var]+parents
    # data = data[cols]
    # try:
    #     n_comp = int((component(data, cols, 'aic') + component(data, cols, 'bic')) / 2)
    #     gmm_model = GaussianMixture(n_components=n_comp).fit(data.values)
    #     gmm_model = GMM(n_components=n_comp, priors=gmm_model.weights_, means=gmm_model.means_, covariances=gmm_model.covariances_)
    #     #gmm_model = GMM(n_components=n_comp).from_samples(data.values)
    #     index = [j for j in range(1, len(parents) + 1)]
    #     ll = 0

    #     if data.shape[1] != 1:
    #         def process_chunk(row):
    #             import numpy as np
    #             # df_chunk.reset_index(inplace=True, drop=True)
    #             #for i in (df_chunk.shape[0]):
    #             # par_val = df_chunk.loc[i, par].values
    #             # v_val = df_chunk.loc[i, v].values
    #             cond_log = np.log(gmm_model.condition(index, row[1:].values).to_probability_density(row[0])[0])
    #             return cond_log
    #         res = data.apply(process_chunk, axis=1)
    #         ll = np.sum(res)


    #     else:

    #         def process_chunk_simple(row):
    #             import numpy as np
    #             # df_chunk.reset_index(inplace=True, drop=True)
    #             #for i in (df_chunk.shape[0]):
    #             # par_val = df_chunk.loc[i, par].values
    #             # v_val = df_chunk.loc[i, v].values
    #             cond_log = np.log(gmm_model.to_probability_density(row[0])[0])
    #             return cond_log
    #         res = data.apply(process_chunk_simple, axis=1)
    #         ll = np.sum(res)
    #     ll = ll - (np.log(data.shape[0]) / 2) * data.shape[1]
    # except:
    #     ll = -100000000
    
    # return round(ll)




def BIC_local_gauss(data, var, parents):
    if len(parents) == 0:
        cols = [var]+parents
        data = data[cols]
        N = data.shape[0]
        d = len(cols)
        n_comp = 1
        gmm_model = GaussianMixture(n_components=n_comp, init_params='k-means++').fit(data.values)
        ll = np.sum(gmm_model.score(data.values)) - 0.007*np.log(N)*n_comp*(d + 1 + d*(d+1)//2)
    else:
        cols = [var]+parents
        data = data[cols]
        N = data.shape[0]
        d = len(cols)
        n_comp = 1
        gmm_model = GaussianMixture(n_components=n_comp, init_params='k-means++').fit(data.values)
        gmm_model = GMM(n_components=n_comp, priors=gmm_model.weights_, means=gmm_model.means_, covariances=gmm_model.covariances_)
        index = [j for j in range(1, len(parents) + 1)]
        cond_m = gmm_model.predict(index, data[parents].values)
        cond_m = cond_m.reshape(N,)
        std = np.abs(data[var].values - cond_m)
        #variance = math.sqrt(mse(data[var].values, cond_m))
        p = norm(loc=cond_m, scale=std).logpdf(data[var].values)
        p = [0 if math.isnan(x) else x for x in p]
        ll = (np.sum(p)/N) - 0.007*np.log(N)*n_comp*(d + 1 + d*(d+1)//2)

    return round(ll)
    # v = []
    # par = []
    # if data.shape[1] == 1:
    #     v.append(data.columns[0])
    # else:
    #     v.append(data.columns[0])
    #     par = data.columns[1:]
    # ll = 0
    # n_comp = 3
    # #gmm_model = GaussianMixture(n_components=n_comp).fit(data.values)
    # #gmm_model = GMM(n_components=n_comp, priors=gmm_model.weights_, means=gmm_model.means_, covariances=gmm_model.covariances_)
    # gmm_model = GMM(n_components=n_comp).from_samples(data.values)

    # if data.shape[1] != 1:
        
    #     index = [j for j in range(1, len(par) + 1)]
    #     for i in range(data.shape[0]):
    #         par_val = data.loc[i, par].values
    #         v_val = data.loc[i, v].values
    #         cond_log = np.log(gmm_model.condition(index, par_val).to_probability_density(v_val)[0])
    #         ll += cond_log
    # else:
    #     for i in range(data.shape[0]):
    #         ll_log = np.log(gmm_model.to_probability_density(data.loc[i,v].values)[0]) 
    #         ll += ll_log
    # ll = ll - (np.log(data.shape[0]) / 2) * data.shape[1]
    
    #return round(ll)



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
        score = BIC_local_gmm(self.data, variable, parents)

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
        score = BIC_local_gauss(self.data, variable, parents)


        return score



class BICG(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for BIC for BayesianModels with MI for Gaussian.
        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        """
        super(BICG, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""
        nrow = len(self.data)
        list_var = [variable]
        list_var.extend(parents)       
        score = - BIC_local(self.data[list_var])

        return score
