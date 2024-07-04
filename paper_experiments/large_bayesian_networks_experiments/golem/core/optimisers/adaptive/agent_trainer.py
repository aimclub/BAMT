import operator
from copy import deepcopy
from functools import reduce
from typing import Sequence, Optional, Any, Tuple, List, Iterable

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.optimisers.adaptive.common_types import TrajectoryStep, GraphTrajectory
from golem.core.optimisers.adaptive.experience_buffer import ExperienceBuffer
from golem.core.optimisers.adaptive.operator_agent import OperatorAgent
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.utilities.data_structures import unzip


class AgentTrainer:
    """Utility class providing fit/validate logic for adaptive Mutation agents.
    Works in tandem with `HistoryReader`.

    How to use offline training:

    1. Collect histories to some directory using `ExperimentLauncher`
    2. Create optimizer & Pretrain mutation agent on these histories using `HistoryReader` and `AgentTrainer`
    3. Optionally, validate the Agent on validation set of histories
    4. Run optimization with pretrained agent
    """

    def __init__(self,
                 objective: Objective,
                 mutation_operator: Mutation,
                 agent: Optional[OperatorAgent] = None,
                 ):
        self._log = default_log(self)
        self.agent = agent if agent is not None else mutation_operator.agent
        self.mutation = mutation_operator
        self.objective = objective
        self._adapter = self.mutation.graph_generation_params.adapter

    def fit(self, histories: Iterable[OptHistory], validate_each: int = -1) -> OperatorAgent:
        """
        Method to fit trainer on collected histories.
        param histories: histories to use in training.
        param validate_each: validate agent once in validate_each generation.
        """
        # Set mutation probabilities to 1.0
        initial_req = deepcopy(self.mutation.requirements)
        self.mutation.requirements.mutation_prob = 1.0

        for i, history in enumerate(histories):
            # Preliminary validity check
            # This allows to filter out histories with different objectives automatically
            if history.objective.metric_names != self.objective.metric_names:
                self._log.warning(f'History #{i+1} has different objective! '
                                  f'Expected {self.objective}, got {history.objective}.')
                continue

            # Build datasets
            experience = ExperienceBuffer.from_history(history)
            val_experience = None
            if validate_each > 0 and i % validate_each == 0:
                experience, val_experience = experience.split(ratio=0.8, shuffle=True)

            # Train
            self._log.info(f'Training on history #{i+1} with {len(history.generations)} generations')
            self.agent.partial_fit(experience)

            # Validate
            if val_experience:
                reward_loss, reward_target = self.validate_agent(experience=val_experience)
                self._log.info(f'Agent validation for history #{i+1} & {experience}: '
                               f'Reward target={reward_target:.3f}, loss={reward_loss:.3f}')

        # Reset mutation probabilities to default
        self.mutation.update_requirements(requirements=initial_req)
        return self.agent

    def validate_on_rollouts(self, histories: Sequence[OptHistory]) -> float:
        """Validates rollouts of agent vs. historic trajectories, comparing
        their mean total rewards (i.e. total fitness gain over the trajectory)."""

        # Collect all trajectories from all histories; and their rewards
        trajectories = concat_lists(map(ExperienceBuffer.unroll_trajectories, histories))

        mean_traj_len = int(np.mean([len(tr) for tr in trajectories]))
        traj_rewards = [sum(reward for _, reward, _ in traj) for traj in trajectories]
        mean_baseline_reward = np.mean(traj_rewards)

        # Collect same number of trajectories of the same length; and their rewards
        agent_trajectories = [self._sample_trajectory(initial=tr[0][0], length=mean_traj_len)
                              for tr in trajectories]
        agent_traj_rewards = [sum(reward for _, reward, _ in traj) for traj in agent_trajectories]
        mean_agent_reward = np.mean(agent_traj_rewards)

        # Compute improvement score of agent over baseline histories
        improvement = mean_agent_reward - mean_baseline_reward
        return improvement

    def validate_history(self, history: OptHistory) -> Tuple[float, float]:
        """Validates history of mutated individuals against optimal policy."""
        history_trajectories = ExperienceBuffer.unroll_trajectories(history)
        return self._validate_against_optimal(history_trajectories)

    def validate_agent(self,
                       graphs: Optional[Sequence[Graph]] = None,
                       experience: Optional[ExperienceBuffer] = None) -> Tuple[float, float]:
        """Validates agent policy against optimal policy on given graphs."""
        if experience:
            agent_steps = experience.retrieve_trajectories()
        elif graphs:
            agent_steps = [self._make_action_step(Individual(g)) for g in graphs]
        else:
            self._log.warning('Either graphs or history must not be None for validation!')
            return 0., 0.
        return self._validate_against_optimal(trajectories=[agent_steps])

    def _validate_against_optimal(self, trajectories: Sequence[GraphTrajectory]) -> Tuple[float, float]:
        """Validates a policy trajectories against optimal policy
        that at each step always chooses the best action with max reward."""
        reward_losses = []
        reward_targets = []
        for trajectory in trajectories:
            inds, actions, rewards = unzip(trajectory)
            _, best_actions, best_rewards = self._apply_best_action(inds)
            reward_loss = self._compute_reward_loss(rewards, best_rewards)
            reward_losses.append(reward_loss)
            reward_targets.append(np.mean(best_rewards))
        reward_loss = float(np.mean(reward_losses))
        reward_target = float(np.mean(reward_targets))
        return reward_loss, reward_target

    @staticmethod
    def _compute_reward_loss(rewards, optimal_rewards, normalized=False) -> float:
        """Returns difference (or deviation) from optimal reward.
        When normalized, 0. means actual rewards match optimal rewards completely,
        0.5 means they on average deviate by 50% from optimal rewards,
        and 2.2 means they on average deviate by more than 2 times from optimal reward."""
        reward_losses = np.subtract(optimal_rewards, rewards)  # always positive
        if normalized:
            reward_losses = reward_losses / np.abs(optimal_rewards) \
                if np.count_nonzero(optimal_rewards) == optimal_rewards.size else reward_losses
        means = np.mean(reward_losses)
        return float(means)

    def _apply_best_action(self, inds: Sequence[Individual]) -> TrajectoryStep:
        """Returns greedily optimal mutation for given graph and associated reward."""
        candidates = []
        for ind in inds:
            for mutation_id in self.agent.available_actions:
                try:
                    values = self._apply_action(mutation_id, ind)
                    candidates.append(values)
                except Exception as e:
                    self._log.warning(f'Eval error for mutation <{mutation_id}> '
                                      f'on graph: {ind.graph.descriptive_id}:\n{e}')
                    continue
        best_step = max(candidates, key=lambda step: step[-1])
        return best_step

    def _apply_action(self, action: Any, ind: Individual) -> TrajectoryStep:
        new_graph, applied = self.mutation._adapt_and_apply_mutation(ind.graph, action)
        fitness = self._eval_objective(new_graph) if applied else None
        parent_op = ParentOperator(type_='mutation', operators=applied, parent_individuals=ind)
        new_ind = Individual(new_graph, fitness=fitness, parent_operator=parent_op)

        prev_fitness = ind.fitness or self._eval_objective(ind.graph)
        if prev_fitness and fitness:
            reward = prev_fitness.value - fitness.value
        elif prev_fitness and not fitness:
            reward = -1.
        else:
            reward = 0.
        return new_ind, action, reward

    def _eval_objective(self, graph: Graph) -> Fitness:
        return self._adapter.adapt_func(self.objective)(graph)

    def _make_action_step(self, ind: Individual) -> TrajectoryStep:
        action = self.agent.choose_action(ind.graph)
        return self._apply_action(action, ind)

    def _sample_trajectory(self, initial: Individual, length: int) -> GraphTrajectory:
        trajectory = []
        past_ind = initial
        for i in range(length):
            next_ind, action, reward = self._make_action_step(past_ind)
            trajectory.append((next_ind, action, reward))
            past_ind = next_ind
        return trajectory


def concat_lists(lists: Iterable[List]) -> List:
    return reduce(operator.add, lists, [])
