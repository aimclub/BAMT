from typing import Optional, Union

from sklearn import linear_model

from .gaussian_node import GaussianNode
from .schema import GaussianParams, HybcprobParams

NodeInfo = Union[GaussianParams, HybcprobParams]


class CompositeContinuousNode(GaussianNode):
    def __init__(self, name, regressor: Optional[object] = None):
        super(CompositeContinuousNode, self).__init__(name)
        if regressor is None:
            regressor = linear_model.LinearRegression()
        self.regressor = regressor
        self.type = "CompositeContinuous" + f" ({type(self.regressor).__name__})"
