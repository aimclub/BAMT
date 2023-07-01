from bamt.networks.base import BaseNetwork
from bamt.log import logger_network
from typing import Dict
import bamt.builders as builders


class CompositeBN(BaseNetwork):
    """
    Composite Bayesian Network with Machine Learning Models support
    """

    def __init__(self):
        super(CompositeBN, self).__init__()
        self._allowed_dtypes = ['cont', 'disc', 'disc_num']
        self.type = 'Composite'
