from .base import BaseNode
from .gaussian_node import GaussianNode
from .conditional_gaussian_node import ConditionalGaussianNode
from .schema import GaussianParams, HybcprobParams

from sklearn import linear_model
from pandas import DataFrame
from typing import Optional, List, Union


NodeInfo = Union[GaussianParams, HybcprobParams]


class CompositeContinuousNode(BaseNode):
    """
    Main class for Composite Continuous Node, combining the functionalities of Gaussian and Conditional Gaussian Nodes
    """

    def __init__(self, name, regressor: Optional[object] = None):
        super(CompositeContinuousNode, self).__init__(name)
        if regressor is None:
            regressor = linear_model.LinearRegression()
        self.regressor = regressor
        self.type = 'CompositeContinuous' + \
            f" ({type(self.regressor).__name__})"

    def fit_parameters(self, data: DataFrame) -> NodeInfo:
        if self.disc_parents:
            return ConditionalGaussianNode(
                self.name, self.regressor).fit_parameters(data)
        else:
            return GaussianNode(self.name, self.regressor).fit_parameters(data)

    def choose(self, node_info: NodeInfo,
               pvals: List[Union[str, float]]) -> float:
        if self.disc_parents:
            return ConditionalGaussianNode(
                self.name, self.regressor).choose(
                node_info.get(
                    'hybcprob', {}), pvals)
        else:
            return GaussianNode(
                self.name,
                self.regressor).choose(
                node_info,
                pvals)

    def predict(self, node_info: NodeInfo,
                pvals: List[Union[str, float]]) -> float:
        if self.disc_parents:
            return ConditionalGaussianNode(
                self.name, self.regressor).predict(
                node_info.get(
                    'hybcprob', {}), pvals)
        else:
            return GaussianNode(
                self.name,
                self.regressor).predict(
                node_info,
                pvals)
