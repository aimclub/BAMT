from typing import Sequence, List

from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.neural_mab import NeuralMAB
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.common_types import ActType


class NeuralContextualMultiArmedBanditAgent(ContextualMultiArmedBanditAgent):
    """ Neural Contextual Multi-Armed bandit.
    Observations can be encoded with the use of Neural Networks, but still there are some restrictions
    to guarantee convergence. """
    def __init__(self,
                 actions: Sequence[ActType],
                 context_agent_type: ContextAgentTypeEnum,
                 available_operations: List[str],
                 n_jobs: int = 1,
                 enable_logging: bool = True,
                 decaying_factor: float = 1.0):
        super().__init__(actions=actions, n_jobs=n_jobs,
                         context_agent_type=context_agent_type, available_operations=available_operations,
                         enable_logging=enable_logging, decaying_factor=decaying_factor)
        self._agent = NeuralMAB(arms=self._indices,
                                n_jobs=n_jobs)
