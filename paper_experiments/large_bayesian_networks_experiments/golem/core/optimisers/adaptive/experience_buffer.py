from collections import deque
from typing import List, Iterable, Tuple, Optional

import numpy as np

from golem.core.optimisers.adaptive.common_types import ObsType, ActType, TrajectoryStep, GraphTrajectory
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


class ExperienceBuffer:
    """Buffer for learning experience of ``OperatorAgent``.
    Keeps (State, Action, Reward) lists until retrieval."""

    def __init__(self, window_size: Optional[int] = None, inds=None, actions=None, rewards=None):
        self.window_size = window_size
        self._prev_pop = set()
        self._next_pop = set()

        if inds and not (len(inds) == len(actions) == len(rewards)):
            raise ValueError('lengths of buffers do not match')
        self._individuals = deque(inds) if inds else deque(maxlen=self.window_size)
        self._actions = deque(actions) if actions else deque(maxlen=self.window_size)
        self._rewards = deque(rewards) if rewards else deque(maxlen=self.window_size)

    @staticmethod
    def from_history(history: OptHistory) -> 'ExperienceBuffer':
        exp = ExperienceBuffer()
        exp.collect_history(history)
        return exp

    def _reset(self):
        self._prev_pop = set()
        self._next_pop = set()

        # if window size was not specified than there is no need to store these values for reuse.
        # Otherwise, if the window_size is specified, then storages will be updated automatically in queue
        if self.window_size is None:
            self._individuals = deque(maxlen=self.window_size)
            self._actions = deque(maxlen=self.window_size)
            self._rewards = deque(maxlen=self.window_size)

    @staticmethod
    def unroll_action_step(result: Individual) -> TrajectoryStep:
        """Unrolls individual's history to get its source individual, action and resulting reward."""
        if not result.parent_operator or result.parent_operator.type_ != 'mutation':
            return None, None, np.nan
        source_ind = result.parent_operator.parent_individuals[0]
        action = result.parent_operator.operators[0]
        # we're minimising the fitness, that's why less is better
        reward = (source_ind.fitness.value - result.fitness.value) / abs(source_ind.fitness.value)\
            if source_ind.fitness and source_ind.fitness.value != 0. else 0.
        return source_ind, action, reward

    @staticmethod
    def unroll_trajectories(history: OptHistory) -> List[GraphTrajectory]:
        """Iterates through history and find continuous sequences of applied operator actions."""
        trajectories = []
        seen_uids = set()
        for terminal_individual in history.final_choices:
            trajectory = []
            next_ind = terminal_individual
            while True:
                seen_uids.add(next_ind.uid)
                source_ind, action, reward = ExperienceBuffer.unroll_action_step(next_ind)
                if source_ind is None or source_ind.uid in seen_uids:
                    break
                # prepend step to keep historical direction
                trajectory.insert(0, (source_ind, action, reward))
                next_ind = source_ind
            trajectories.append(trajectory)
        return trajectories

    def collect_history(self, history: OptHistory):
        seen = set()
        # We don't need the initial assumptions, as they have no parent operators, hence [1:]
        for generation in history.generations[1:]:
            for ind in generation:
                if ind.uid not in seen:
                    seen.add(ind.uid)
                    self.collect_result(ind)

    def collect_results(self, results: Iterable[Individual]):
        for ind in results:
            self.collect_result(ind)

    def collect_result(self, result: Individual):
        if result.uid in self._prev_pop:
            # avoid collecting results from individuals that didn't change
            return
        self._next_pop.add(result.uid)

        source_ind, action, reward = self.unroll_action_step(result)
        if action is None:
            return
        self.collect_experience(source_ind, action, reward)

    def collect_experience(self, obs: Individual, action: ActType, reward: float):
        self._individuals.append(obs)
        self._actions.append(action)
        self._rewards.append(reward)

    def retrieve_experience(self, as_graphs: bool = True) -> Tuple[List[ObsType], List[ActType], List[float]]:
        """Get all collected experience and clear the experience buffer.
        Args:
            as_graphs: if True (by default) returns observations as graphs, otherwise as individuals.
        Return:
             Unzipped trajectories (tuple of lists of observations, actions, rewards).
        """
        individuals, actions, rewards = self._individuals, self._actions, self._rewards
        observations = [ind.graph for ind in individuals] if as_graphs else individuals
        next_pop = self._next_pop
        self._reset()
        self._prev_pop = next_pop
        return list(observations), list(actions), list(rewards)

    def retrieve_trajectories(self) -> GraphTrajectory:
        """Same as `retrieve_experience` but in the form of zipped trajectories that consist from steps."""
        trajectories = list(zip(*self.retrieve_experience(as_graphs=False)))
        return trajectories

    def split(self, ratio: float = 0.8, shuffle: bool = False
              ) -> Tuple['ExperienceBuffer', 'ExperienceBuffer']:
        """Splits buffer in 2 parts, useful for train/validation split."""
        mask_train = np.full_like(self._individuals, False, dtype=bool)
        num_train = int(len(self._individuals) * ratio)
        mask_train[-num_train:] = True
        if shuffle:
            np.random.default_rng().shuffle(mask_train)
        buffer_train = ExperienceBuffer(inds=np.array(self._individuals)[mask_train].tolist(),
                                        actions=np.array(self._actions)[mask_train].tolist(),
                                        rewards=np.array(self._rewards)[mask_train].tolist())
        buffer_val = ExperienceBuffer(inds=np.array(self._individuals)[~mask_train].tolist(),
                                      actions=np.array(self._actions)[~mask_train].tolist(),
                                      rewards=np.array(self._rewards)[~mask_train].tolist())
        return buffer_train, buffer_val

    def __len__(self):
        return len(self._individuals)

    def __str__(self):
        return f'{self.__class__.__name__}({len(self)})'
