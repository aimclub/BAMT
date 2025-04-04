import itertools
from typing import Union, List, Optional, Dict
import numpy as np
from sklearn.mixture import GaussianMixture
from pandas import DataFrame
from scipy.stats import multivariate_normal

from bamt.utils.MathUtils import component
from bamt.nodes.base import BaseNode
from bamt.nodes.schema import CondMixtureGaussParams


class ConditionalMixtureGaussianNode(BaseNode):
    """
    Main class for Conditional Mixture Gaussian Node using scikit-learn's GaussianMixture
    """
    def __init__(self, name):
        super(ConditionalMixtureGaussianNode, self).__init__(name)
        self.type = "ConditionalMixtureGaussian"

    def fit_parameters(self, data: DataFrame) -> Dict[str, Dict[str, CondMixtureGaussParams]]:
        """Train params for Conditional Mixture Gaussian Node"""
        hycprob = dict()
        values = []
        combinations = []
        
        # Get all possible values of discrete parents
        for d_p in self.disc_parents:
            values.append(np.unique(data[d_p].values))
            
        # Generate all combinations of discrete parent values
        for xs in itertools.product(*values):
            combinations.append(list(xs))
            
        # For each combination, fit a GMM
        for comb in combinations:
            # Filter data based on discrete parent values
            mask = np.full(len(data), True)
            for col, val in zip(self.disc_parents, comb):
                mask = (mask) & (data[col] == val)
                
            new_data = data[mask]
            new_data.reset_index(inplace=True, drop=True)
            key_comb = [str(x) for x in comb]
            nodes = [self.name] + self.cont_parents
            
            if new_data.shape[0] > 5:
                if self.cont_parents:
                    # Determine number of components
                    n_comp = int(
                        (component(new_data, nodes, "aic") + 
                         component(new_data, nodes, "bic")) / 2
                    )
                    
                    # Fit sklearn's GaussianMixture
                    gmm = GaussianMixture(
                        n_components=n_comp,
                        max_iter=500,
                        init_params='kmeans'
                    ).fit(new_data[nodes].values)
                else:
                    # Determine number of components
                    n_comp = int(
                        (component(new_data, [self.name], "aic") + 
                         component(new_data, [self.name], "bic")) / 2
                    )
                    
                    # Fit sklearn's GaussianMixture
                    gmm = GaussianMixture(
                        n_components=n_comp,
                        max_iter=500,
                        init_params='kmeans'
                    ).fit(np.transpose([new_data[self.name].values]))
                
                # Extract parameters
                means = gmm.means_.tolist()
                cov = gmm.covariances_.tolist()
                w = gmm.weights_.tolist()
                
                hycprob[str(key_comb)] = {"covars": cov, "mean": means, "coef": w}
                
            elif new_data.shape[0] != 0:
                n_comp = 1
                
                # Fit one-component GMM
                if self.cont_parents:
                    gmm = GaussianMixture(
                        n_components=n_comp
                    ).fit(new_data[nodes].values)
                else:
                    gmm = GaussianMixture(
                        n_components=n_comp
                    ).fit(np.transpose([new_data[self.name].values]))
                
                # Extract parameters
                means = gmm.means_.tolist()
                cov = gmm.covariances_.tolist()
                w = gmm.weights_.tolist()
                
                hycprob[str(key_comb)] = {"covars": cov, "mean": means, "coef": w}
            else:
                # Handle empty data
                hycprob[str(key_comb)] = {"covars": np.nan, "mean": np.nan, "coef": []}
                
        return {"hybcprob": hycprob}

    @staticmethod
    def get_dist(node_info, pvals):
        lgpvals = []
        dispvals = []
        
        # Separate discrete and continuous parent values
        for pval in pvals:
            if (isinstance(pval, str)) | (isinstance(pval, int)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
        
        # Get distribution for this combination of discrete parent values
        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        covariance = lgdistribution["covars"]
        w = lgdistribution["coef"]
        
        if len(w) != 0:
            if len(lgpvals) != 0:
                indices = [i for i in range(1, (len(lgpvals) + 1), 1)]
                if not np.isnan(np.array(lgpvals)).all():
                    # Implement conditioning directly
                    n_features = len(mean[0])
                    n_comp = len(w)
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
                            cond_means[k] = mean_b + cov_ba.dot(cov_aa_inv).dot(lgpvals - mean_a)
                            cond_covs[k] = cov_bb - cov_ba.dot(cov_aa_inv).dot(cov_ab)
                            
                            # Compute marginal probability
                            marginal_probs[k] = w[k] * multivariate_normal.pdf(
                                lgpvals, mean=mean_a, cov=cov_aa)
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

    def choose(self, node_info: Dict[str, Dict[str, CondMixtureGaussParams]], 
              pvals: List[Union[str, float]]) -> Optional[float]:
        """Sample a value from the node"""
        mean, covariance, w = self.get_dist(node_info, pvals)
        
        # Check if we have a valid distribution
        if not isinstance(w, np.ndarray) and not isinstance(w, list):
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
    def predict(node_info: Dict[str, Dict[str, CondMixtureGaussParams]],
               pvals: List[Union[str, float]]) -> Optional[float]:
        """Predict a value from the node"""
        lgpvals = []
        dispvals = []
        
        # Separate discrete and continuous parent values
        for pval in pvals:
            if (isinstance(pval, str)) | (isinstance(pval, int)):
                dispvals.append(pval)
            else:
                lgpvals.append(pval)
                
        # Get distribution for this combination of discrete parent values
        lgdistribution = node_info["hybcprob"][str(dispvals)]
        mean = lgdistribution["mean"]
        covariance = lgdistribution["covars"]
        w = lgdistribution["coef"]
        
        if len(w) != 0:
            if len(lgpvals) != 0:
                indices = [i for i in range(1, (len(lgpvals) + 1), 1)]
                if not np.isnan(np.array(lgpvals)).all():
                    # Implement prediction directly
                    n_features = len(mean[0])
                    n_comp = len(w)
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
                            component_pred = mean_b + beta.dot(np.array(lgpvals) - mean_a)
                            
                            # Compute marginal probability for weighting
                            weight = w[k] * multivariate_normal.pdf(
                                lgpvals, mean=mean_a, cov=cov_aa)
                            
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
                # Compute weighted mean when no continuous parents
                sample = 0
                for ind, wi in enumerate(w):
                    sample += wi * mean[ind][0]
                return sample
        else:
            return np.nan
