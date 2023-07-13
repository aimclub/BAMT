from bamt.networks.base import BaseNetwork
from bamt.log import logger_network
import pandas as pd
import numpy as np
from typing import Optional
from bamt.builders.builders_base import ParamDict
from bamt.builders.composite_builder import CompositeStructureBuilder


class CompositeBN(BaseNetwork):
    """
    Composite Bayesian Network with Machine Learning Models support
    """

    def __init__(self):
        super(CompositeBN, self).__init__()
        self._allowed_dtypes = ['cont', 'disc', 'disc_num']
        self.type = 'Composite'

    def add_edges(self,
                  data: pd.DataFrame,
                  progress_bar: bool = True,
                  classifier: Optional[object] = None,
                  regressor: Optional[object] = None,
                  **kwargs):

        worker = CompositeStructureBuilder(
            data=data,
            descriptor=self.descriptor,
            regressor=regressor)

        worker.build(
            data=data,
            classifier=classifier,
            regressor=regressor,
            progress_bar=progress_bar,
            **kwargs)

        # update family
        self.nodes = worker.skeleton['V']
        self.edges = worker.skeleton['E']
