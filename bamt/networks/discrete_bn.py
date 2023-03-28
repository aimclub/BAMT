from .base import BaseNetwork


class DiscreteBN(BaseNetwork):
    """
    Bayesian Network with Discrete Types of Nodes
    """

    def __init__(self):
        super(DiscreteBN, self).__init__()
        self.type = 'Discrete'
        self.scoring_function = ""
        self._allowed_dtypes = ['disc', 'disc_num']
        self.has_logit = None
        self.use_mixture = None
