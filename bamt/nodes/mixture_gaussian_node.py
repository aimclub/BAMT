import numpy as np

from .base import BaseNode
from .schema import MixtureGaussianParams

from pandas import DataFrame
from bamt.utils.MathUtils import component
from gmr import GMM

from typing import Union, List, Optional


class MixtureGaussianNode(BaseNode):
    """
    Main class for Mixture Gaussian Node
    """

    def __init__(self, name):
        super(MixtureGaussianNode, self).__init__(name)
        self.type = 'MixtureGaussian'

    def fit_parameters(self, data: DataFrame) -> MixtureGaussianParams:
        """
        Train params for Mixture Gaussian Node
        """
        parents = self.disc_parents + self.cont_parents
        if not parents:
            n_comp = int((component(data,
                                    [self.name],
                                    'aic') + component(data,
                                                       [self.name],
                                                       'bic')) / 2)  # component(data, [node], 'LRTS')#
            # n_comp = 3
            gmm = GMM(n_components=n_comp).from_samples(np.transpose(
                [data[self.name].values]), n_iter=500, init_params='kmeans++')
            means = gmm.means.tolist()
            cov = gmm.covariances.tolist()
            # weigts = np.transpose(gmm.to_responsibilities(np.transpose([data[node].values])))
            w = gmm.priors.tolist()  # []
            # for row in weigts:
            #     w.append(np.mean(row))
            return {"mean": means, "coef": w, "covars": cov}
        if parents:
            if not self.disc_parents and self.cont_parents:
                nodes = [self.name] + self.cont_parents
                new_data = data[nodes]
                new_data.reset_index(inplace=True, drop=True)
                n_comp = int(
                    (component(
                        new_data,
                        nodes,
                        'aic') +
                     component(
                         new_data,
                         nodes,
                         'bic')) /
                    2)  # component(new_data, nodes, 'LRTS')#
                # n_comp = 3
                gmm = GMM(
                    n_components=n_comp).from_samples(
                    new_data[nodes].values,
                    n_iter=500,
                    init_params='kmeans++')
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(new_data[nodes].values))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                return {"mean": means,
                        "coef": w,
                        "covars": cov}

    @staticmethod
    def choose(node_info: MixtureGaussianParams,
               pvals: List[Union[str, float]]) -> Optional[float]:
        """
        Func to get value from current node
        node_info: nodes info from distributions
        pvals: parent values
        Return value from MixtureGaussian node
        """
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info['coef'])
        if n_comp != 0:
            if pvals:
                indexes = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    gmm = GMM(
                        n_components=n_comp,
                        priors=w,
                        means=mean,
                        covariances=covariance)
                    cond_gmm = gmm.condition(indexes, [pvals])
                    sample = cond_gmm.sample(1)[0][0]
                else:
                    sample = np.nan
            else:
                gmm = GMM(
                    n_components=n_comp,
                    priors=w,
                    means=mean,
                    covariances=covariance)
                sample = gmm.sample(1)[0][0]
        else:
            sample = np.nan
        return sample

    @staticmethod
    def predict(node_info: MixtureGaussianParams,
                pvals: List[Union[str, float]]) -> Optional[float]:
        """
        Func to get prediction from current node
        node_info: nodes info from distributions
        pvals: parent values
        Return value from MixtureGaussian node
        """
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info['coef'])
        if n_comp != 0:
            if pvals:
                indexes = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    gmm = GMM(
                        n_components=n_comp,
                        priors=w,
                        means=mean,
                        covariances=covariance)
                    sample = gmm.predict(indexes, [pvals])[0][0]
                else:
                    sample = np.nan
            else:
                # gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=covariance)
                sample = 0
                for ind, wi in enumerate(w):
                    sample += wi * mean[ind][0]
        else:
            sample = np.nan
        return sample
