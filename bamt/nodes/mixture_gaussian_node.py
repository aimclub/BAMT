from typing import Union, List, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from pandas import DataFrame
from scipy.stats import multivariate_normal

from bamt.utils.MathUtils import component
from bamt.nodes.base import BaseNode
from bamt.nodes.schema import MixtureGaussianParams


class MixtureGaussianNode(BaseNode):
    """
    Main class for Mixture Gaussian Node using scikit-learn's GaussianMixture
    """
    def __init__(self, name):
        super(MixtureGaussianNode, self).__init__(name)
        self.type = "MixtureGaussian"

    def fit_parameters(self, data: DataFrame) -> MixtureGaussianParams:
        """Train params for Mixture Gaussian Node"""
        parents = self.disc_parents + self.cont_parents
        
        if not parents:
            # Determine number of components using AIC and BIC
            n_comp = int(
                (component(data, [self.name], "aic") + component(data, [self.name], "bic")) / 2
            )
            
            # Fit sklearn's GaussianMixture
            X = np.transpose([data[self.name].values])
            gmm = GaussianMixture(
                n_components=n_comp, 
                max_iter=500, 
                init_params='kmeans'
            ).fit(X)
            
            # Extract parameters
            means = gmm.means_.tolist()
            cov = gmm.covariances_.tolist()
            w = gmm.weights_.tolist()
            
            return {"mean": means, "coef": w, "covars": cov}
            
        if not self.disc_parents and self.cont_parents:
            nodes = [self.name] + self.cont_parents
            new_data = data[nodes]
            new_data.reset_index(inplace=True, drop=True)
            
            # Determine number of components using AIC and BIC
            n_comp = int(
                (component(new_data, nodes, "aic") + component(new_data, nodes, "bic")) / 2
            )
            
            # Fit sklearn's GaussianMixture
            X = new_data[nodes].values
            gmm = GaussianMixture(
                n_components=n_comp, 
                max_iter=500, 
                init_params='kmeans'
            ).fit(X)
            
            # Extract parameters
            means = gmm.means_.tolist()
            cov = gmm.covariances_.tolist()
            w = gmm.weights_.tolist()
            
            return {"mean": means, "coef": w, "covars": cov}

    @staticmethod
    def get_dist(node_info, pvals):
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info["coef"])
        
        if n_comp != 0:
            if pvals:
                indices = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    # Implement conditioning directly
                    # Get remaining indices
                    n_features = len(mean[0])
                    remain_indices = [i for i in range(n_features) if i not in indices]
                    
                    # Initialize parameters for conditional distribution
                    cond_means = np.zeros((n_comp, len(remain_indices)))
                    cond_covs = np.zeros((n_comp, len(remain_indices), len(remain_indices)))
                    marginal_probs = np.zeros(n_comp)
                    
                    # For each component, compute conditional distribution
                    for k in range(n_comp):
                        # Split mean and covariance for conditioning
                        mean_a = np.array([mean[k][i] for i in indices])
                        mean_b = np.array([mean[k][i] for i in remain_indices])
                        
                        cov_aa = np.array([[covariance[k][i][j] for j in indices] for i in indices])
                        cov_ab = np.array([[covariance[k][i][j] for j in remain_indices] for i in indices])
                        cov_ba = np.array([[covariance[k][i][j] for j in indices] for i in remain_indices])
                        cov_bb = np.array([[covariance[k][i][j] for j in remain_indices] for i in remain_indices])
                        
                        # Compute conditional mean and covariance
                        try:
                            cov_aa_inv = np.linalg.inv(cov_aa)
                            cond_means[k] = mean_b + cov_ba.dot(cov_aa_inv).dot(pvals - mean_a)
                            cond_covs[k] = cov_bb - cov_ba.dot(cov_aa_inv).dot(cov_ab)
                            
                            # Compute marginal probability
                            marginal_probs[k] = w[k] * multivariate_normal.pdf(
                                pvals, mean=mean_a, cov=cov_aa)
                        except np.linalg.LinAlgError:
                            # Handle numerical instability
                            cond_means[k] = mean_b
                            cond_covs[k] = cov_bb
                            marginal_probs[k] = w[k] * 1e-10
                    
                    # Normalize weights
                    if np.sum(marginal_probs) > 0:
                        cond_weights = marginal_probs / np.sum(marginal_probs)
                    else:
                        cond_weights = np.ones(n_comp) / n_comp
                    
                    return cond_means, cond_covs, cond_weights
                else:
                    return np.nan, np.nan, np.nan
            else:
                return mean, covariance, w
        else:
            return np.nan, np.nan, np.nan

    def choose(self, node_info: MixtureGaussianParams, pvals: List[Union[str, float]]) -> Optional[float]:
        """Sample a value from the node"""
        mean, covariance, w = self.get_dist(node_info, pvals)
        
        if isinstance(mean, float) and np.isnan(mean):
            return np.nan
            
        n_comp = len(w)
        
        # Sample directly without creating a wrapper
        # First, choose a component based on weights
        component_idx = np.random.choice(n_comp, p=w)
        
        # Then sample from the selected component
        if mean.ndim == 3:  # Handle case where mean is 3D array
            sample = multivariate_normal.rvs(
                mean=mean[component_idx][0], 
                cov=covariance[component_idx][0][0]
            )
        else:
            sample = multivariate_normal.rvs(
                mean=mean[component_idx], 
                cov=covariance[component_idx]
            )
            
        if isinstance(sample, np.ndarray):
            return sample[0]
        return sample

    @staticmethod
    def predict(node_info: MixtureGaussianParams, pvals: List[Union[str, float]]) -> Optional[float]:
        """Predict a value from the node"""
        mean = node_info["mean"]
        covariance = node_info["covars"]
        w = node_info["coef"]
        n_comp = len(node_info["coef"])
        
        if n_comp != 0:
            if pvals:
                indices = [i for i in range(1, len(pvals) + 1)]
                if not np.isnan(np.array(pvals)).all():
                    # Implement prediction directly
                    n_features = len(mean[0])
                    output_indices = [i for i in range(n_features) if i not in indices]
                    
                    # For each component, compute regression
                    predictions = np.zeros(len(output_indices))
                    total_weight = 0
                    
                    for k in range(n_comp):
                        # Split parameters for regression
                        mean_a = np.array([mean[k][i] for i in indices])
                        mean_b = np.array([mean[k][i] for i in output_indices])
                        
                        cov_aa = np.array([[covariance[k][i][j] for j in indices] for i in indices])
                        cov_ab = np.array([[covariance[k][i][j] for j in output_indices] for i in indices])
                        
                        # Compute regression coefficients
                        try:
                            cov_aa_inv = np.linalg.inv(cov_aa)
                            beta = cov_ab.T.dot(cov_aa_inv)
                            
                            # Compute prediction for this component
                            component_pred = mean_b + beta.dot(np.array(pvals) - mean_a)
                            
                            # Compute marginal probability for weighting
                            weight = w[k] * multivariate_normal.pdf(
                                pvals, mean=mean_a, cov=cov_aa)
                            
                            predictions += weight * component_pred
                            total_weight += weight
                        except np.linalg.LinAlgError:
                            continue
                    
                    if total_weight > 0:
                        predictions /= total_weight
                        return predictions[0]
                    else:
                        return np.nan
                else:
                    return np.nan
            else:
                # Compute weighted mean
                sample = 0
                for ind, wi in enumerate(w):
                    sample += wi * mean[ind][0]
                return sample
        else:
            return np.nan
