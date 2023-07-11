from .base import BaseNetwork


class ContinuousBN(BaseNetwork):
    """
    Bayesian Network with Continuous Types of Nodes
    """

    def __init__(self, use_mixture: bool = False):
        super(ContinuousBN, self).__init__()
        self.type = "Continuous"
        self._allowed_dtypes = ["cont"]
        self.has_logit = None
        self.use_mixture = use_mixture
        self.scoring_function = ""
