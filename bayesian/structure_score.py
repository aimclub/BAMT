# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
from bayesian.mi_entropy_gauss import mi_gauss
from bayesian.redef_info_scores import log_lik_local, BIC_local, AIC_local
from pgmpy.estimators.StructureScore import StructureScore

class MIG(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for Bayesian structure scoring for BayesianModels with MI for Gaussian.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        """
        super(MIG, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""
        nrow = len(self.data)
        list_var = [variable]
        list_var.extend(parents)       
        score = - nrow * mi_gauss(self.data[list_var])

        return score

class LLG(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for log-likelihood for BayesianModels with MI for Gaussian.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        """
        super(LLG, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""
        nrow = len(self.data)
        list_var = [variable]
        list_var.extend(parents)       
        score = - log_lik_local(self.data[list_var])

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

class AICG(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for AIC for BayesianModels with MI for Gaussian.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        """
        super(AICG, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""
        nrow = len(self.data)
        list_var = [variable]
        list_var.extend(parents)       
        score = - AIC_local(self.data[list_var])

        return score


    
   
    