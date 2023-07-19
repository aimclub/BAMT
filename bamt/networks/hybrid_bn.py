from .base import BaseNetwork

from typing import Dict


class HybridBN(BaseNetwork):
    """
    Bayesian Network with Mixed Types of Nodes
    """

    def __init__(self, has_logit: bool = False, use_mixture: bool = False):
        super(HybridBN, self).__init__()
        self._allowed_dtypes = ["cont", "disc", "disc_num"]
        self.type = "Hybrid"
        self.has_logit = has_logit
        self.use_mixture = use_mixture

    def validate(self, descriptor: Dict[str, Dict[str, str]]) -> bool:
        types = descriptor["types"]
        s = set(types.values())
        return (
            True
            if ({"cont", "disc", "disc_num"} == s)
            or ({"cont", "disc"} == s)
            or ({"cont", "disc_num"} == s)
            else False
        )
