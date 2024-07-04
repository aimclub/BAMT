from typing import Union, Hashable, Tuple, Sequence

from golem.core.dag.graph import Graph
from golem.core.optimisers.opt_history_objects.individual import Individual

ObsType = Union[Individual, Graph]
ActType = Hashable
# Trajectory step includes (past observation, action, reward)
TrajectoryStep = Tuple[Individual, ActType, float]
# Trajectory is a sequence of applied mutations and received rewards
GraphTrajectory = Sequence[TrajectoryStep]
