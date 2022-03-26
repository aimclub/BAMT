from pgmpy.estimators.StructureScore import StructureScore
from bamt.utils.MathUtils import *
from sklearn.mixture import GaussianMixture
from gmr import GMM
import numpy as np
from scipy.stats import norm
import os,sys,inspect

def BIC_local(data):
    # v = []
    # par = []
    # if len(data.shape) == 0:
    #     v.append(data.columns[0])
    # else:
    #     v.append(data.columns[0])
    #     par = data.columns[1:]
    # n_comp = int((component(data, [], 'AIC') + component(data, [], 'BIC')) / 2)
    # gmm_model = GMM(n_components=n_comp).from_samples(data)
    # ll = 0

    # if data.shape[1] != 1:
    #     for i in range(data.shape[0]):
    #         par_val = data[i,1:]
    #         gmm_cond = gmm_model.condition([j for j in range(1, len(par_val) + 1)], par_val)
    #         ll_log = np.log(gmm_cond.to_probability_density(data[i,0])[0])
    #         ll += ll_log
    # else:
    #     for i in range(data.shape[0]):
    #         ll_log = np.log(gmm_model.to_probability_density(data[i,0])[0])
    #         ll += ll_log
    MI = 0
    if data.shape[1] == 1:
        MI = gmm_entropy_1d(data.values)
    else:
        data_x = data.values[:,(0,)]
        data_y = data.values[:,1:]
        Hx = gmm_entropy_1d(data_x)
        Hy = 0
        Hxy = gmm_entropy_nd(data.values)
        if data.shape[1] - 1 > 1:
            Hy = gmm_entropy_nd(data_y)
        else:
            Hy = gmm_entropy_1d(data_y)
        MI = Hx + Hy - Hxy
        #MI = min(Hx, Hy, MI)



    return MI


def gmm_entropy_1d(data):
    n_comp = int((component(data, [], 'AIC') + component(data, [], 'BIC')) / 2)
    gmm_model = GMM(n_components=n_comp).from_samples(data)
    #gmm_model = GaussianMixture(n_components=n_comp).fit(data)
    entropy_sum = 0
    for i in range(n_comp):
        entropy_sum += (gmm_model.priors[i]) * entropy_gauss_1d(gmm_model.covariances[i][0][0])
    return entropy_sum

def gmm_entropy_nd(data):
    n_comp = int((component(data, [], 'AIC') + component(data, [], 'BIC')) / 2)
    gmm_model = GMM(n_components=n_comp).from_samples(data)
    #gmm_model = GaussianMixture(n_components=n_comp).fit(data)
    entropy_sum = 0
    for i in range(n_comp):
        entropy_sum += (gmm_model.priors[i]) * entropy_gauss_nd(gmm_model.covariances[i])
    return entropy_sum






def entropy_gauss_1d(var):
    if var > 1e-16:
        return 0.5 * (1 + math.log(var*2*math.pi))
    else:
        return sys.float_info.min
def entropy_gauss_nd(cov):
    var = np.linalg.det(cov)
    N = var.ndim
    if var > 1e-16:
        return 0.5 * (N * (1 + math.log(2*math.pi)) + math.log(var))
    else:
        return sys.float_info.min


# def BIC_local(data):
#     n_comp = int((component(data, [], 'AIC') + component(data, [], 'BIC')) / 2)
#     gmm_model = GaussianMixture(n_components=n_comp).fit(data)
#     ll_score = gmm_model.lower_bound_
#     NROW = data.shape[0]
#     k = 1
#     if len(data.shape) > 1:
#         k = data.shape[1]
#     num_params = (k*(k+3)) / 2
#     penalty = 0.5 *(n_comp - 1 + n_comp*num_params)* np.log(NROW) 
#     bic_score = ll_score - penalty
#     return bic_score






class LLGMM(StructureScore):
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
        super(LLGMM, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""
        list_var = [variable]
        list_var.extend(parents)       
        score = log_lik_local(self.data[list_var], variable, parents)
        print(variable, parents)
        print(score)
        print('------------------------')

        return score


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



