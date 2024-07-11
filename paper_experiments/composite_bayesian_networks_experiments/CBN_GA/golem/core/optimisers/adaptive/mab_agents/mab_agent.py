import os.path
import _pickle as pickle
import random
import re
from typing import Union, Sequence, Optional

from mabwiser.mab import MAB, LearningPolicy
from scipy.special import softmax

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.adaptive.operator_agent import OperatorAgent, ActType, ObsType, ExperienceBuffer
from golem.core.optimisers.adaptive.reward_agent import FitnessRateRankRewardTransformer
from golem.core.paths import default_data_dir


class MultiArmedBanditAgent(OperatorAgent):
    def __init__(self,
                 actions: Sequence[ActType],
                 n_jobs: int = 1,
                 enable_logging: bool = True,
                 decaying_factor: float = 1.0,
                 path_to_save: Optional[str] = None,
                 is_initial_fit: bool = True):
        super().__init__(enable_logging)
        self.actions = list(actions)
        self._indices = list(range(len(actions)))
        self._arm_by_action = dict(zip(actions, self._indices))
        self._agent = MAB(arms=self._indices,
                          learning_policy=LearningPolicy.UCB1(alpha=1.25),
                          n_jobs=n_jobs)
        self._reward_agent = FitnessRateRankRewardTransformer(decaying_factor=decaying_factor)
        if is_initial_fit:
            self._initial_fit()
        self._path_to_save = path_to_save

    def _initial_fit(self):
        n = len(self.actions)
        uniform_rewards = [1. / n] * n
        self._agent.fit(decisions=self._indices, rewards=uniform_rewards)

    def choose_action(self, obs: ObsType) -> ActType:
        arm = self._agent.predict()
        action = self.actions[arm]
        return action

    def get_action_values(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        prob_dict = self._agent.predict_expectations()
        prob_list = [prob_dict[i] for i in range(len(prob_dict))]
        return prob_list

    def get_action_probs(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return softmax(self.get_action_values())

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        """Continues learning of underlying agent with new experience."""
        _, arms, processed_rewards = self._get_experience(experience)
        self._agent.partial_fit(decisions=arms, rewards=processed_rewards)

    def _get_experience(self, experience: ExperienceBuffer):
        """ Get experience from ExperienceBuffer, process rewards and log. """
        obs, actions, rewards = experience.retrieve_experience()
        arms = [self._arm_by_action[action] for action in actions]
        processed_rewards = self._reward_agent.get_rewards_for_arms(rewards, arms)
        self._dbg_log(obs, actions, processed_rewards)
        return obs, arms, processed_rewards

    def save(self, path_to_save: Optional[str] = None):
        """ Saves bandit to specified file. """

        if not path_to_save:
            path_to_save = self._path_to_save

        # if path was not specified at all
        if not path_to_save:
            path_to_save = os.path.join(default_data_dir(), 'MAB')

        if not path_to_save.endswith('.pkl'):
            os.makedirs(path_to_save, exist_ok=True)
            mabs_num = [int(name.split('_')[0]) for name in os.listdir(path_to_save)
                        if re.fullmatch(r'\d_mab.pkl', name)]
            if not mabs_num:
                max_saved_mab = 0
            else:
                max_saved_mab = max(mabs_num) + 1
            path_to_file = os.path.join(path_to_save, f'{max_saved_mab}_mab.pkl')
        else:
            path_to_dir = os.path.dirname(path_to_save)
            os.makedirs(path_to_dir, exist_ok=True)
            path_to_file = path_to_save
        with open(path_to_file, 'wb') as f:
            pickle.dump(self, f)
        self._log.info(f"MAB was saved to {path_to_file}")

    @staticmethod
    def load(path: str):
        """ Loads bandit from the specified file. """
        with open(path, 'rb') as f:
            mab = pickle.load(f)
        return mab
