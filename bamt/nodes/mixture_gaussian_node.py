from typing import Union, List, Optional

import numpy as np
#from gmr import GMM
from bamt.utils.gmm_wrapper import GMM

from pandas import DataFrame
from bamt.result_models.node_result import MixtureGaussianNodeResult

from bamt.utils.MathUtils import component
from .base import BaseNode
from .schema import MixtureGaussianParams


class MixtureGaussianNode(BaseNode):
    """
    Main class for Mixture Gaussian Node
    """

    def __init__(self, name):
        super(MixtureGaussianNode, self).__init__(name)
        self.type = "MixtureGaussian"

    def fit_parameters(self, data: DataFrame) -> MixtureGaussianParams:
        """
        Train params for Mixture Gaussian Node
        """
        parents = self.disc_parents + self.cont_parents
        if not parents:
            n_comp = int(
                (
                    component(data, [self.name], "aic")
                    + component(data, [self.name], "bic")
                )
                / 2
            )  # component(data, [node], 'LRTS')#
            # n_comp = 3
            gmm = GMM(n_components=n_comp).from_samples(
                np.transpose([data[self.name].values]),
                n_iter=500,
                init_params="kmeans++",
            )
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
                    (
                        component(new_data, nodes, "aic")
                        + component(new_data, nodes, "bic")
                    )
                    / 2
                )  # component(new_data, nodes, 'LRTS')#
                # n_comp = 3
                gmm = GMM(n_components=n_comp).from_samples(
                    new_data[nodes].values, n_iter=500, init_params="kmeans++"
                )
                means = gmm.means.tolist()
                cov = gmm.covariances.tolist()
                # weigts = np.transpose(gmm.to_responsibilities(new_data[nodes].values))
                w = gmm.priors.tolist()  # []
                # for row in weigts:
                #     w.append(np.mean(row))
                return {"mean": means, "coef": w, "covars": cov}

    @staticmethod
    def get_dist(node_info, pvals):
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info["coef"])
        if n_comp != 0:
            if pvals:
                indexes = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    gmm = GMM(
                        n_components=n_comp,
                        priors=w,
                        means=mean,
                        covariances=covariance,
                    )
                    cond_gmm = gmm.condition(indexes, [pvals])
                    means, covars, priors = cond_gmm.means, cond_gmm.covariances, cond_gmm.priors
                else:
                    means, covars, priors =  np.nan, np.nan, np.nan
            else:
                gmm = GMM(
                    n_components=n_comp, priors=w, means=mean, covariances=covariance
                )
                means, covars, priors =  gmm.means, gmm.covariances, gmm.priors
        else:
            means, covars, priors =  np.nan, np.nan, np.nan

        return MixtureGaussianNodeResult((means, covars, priors), n_components=n_comp)

    def choose(
        self, node_info: MixtureGaussianParams, pvals: List[Union[str, float]]
    ) -> Optional[float]:
        """
        Func to get value from current node
        node_info: nodes info from distributions
        pvals: parent values
        Return value from MixtureGaussian node
        """
        mean, covariance, w = self.get_dist(node_info, pvals).get()

        n_comp = len(w)

        gmm = GMM(
            n_components=n_comp,
            priors=w,
            means=mean,
            covariances=covariance,
        )
        return gmm.sample(1)[0][0]

    @staticmethod
    def predict(
        node_info: MixtureGaussianParams, pvals: List[Union[str, float]]
    ) -> Optional[float]:
        """
        Func to get prediction from current node
        node_info: nodes info from distributions
        pvals: parent values
        Return value from MixtureGaussian node
        """
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info["coef"])
        if n_comp != 0:
            if pvals:
                indexes = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    gmm = GMM(
                        n_components=n_comp,
                        priors=w,
                        means=mean,
                        covariances=covariance,
                    )
                    pred = gmm.predict_conditioned(indexes, [pvals])
                    sample = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)

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
