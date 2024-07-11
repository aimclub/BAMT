from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from golem.core.log import default_log
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.opt_history_objects.individual import Individual

if TYPE_CHECKING:
    from golem.core.optimisers.optimizer import AlgorithmParameters

PopulationT = Sequence[Individual]
EvaluationOperator = Callable[[PopulationT], PopulationT]


class Operator(ABC):
    """ Base abstract functional interface for genetic operators.
    Specific signatures are:
    - Selection: Population -> Population
    - Inheritance: [Population, Population] -> Population
    - Regularization: [Population, EvaluationOperator] -> Population
    - Mutation: Union[Individual, Population] -> Union[Individual, Population]
    - Crossover: Population -> Population
    - Elitism: [Population, Population] -> Population
    """

    def __init__(self,
                 parameters: Optional['AlgorithmParameters'] = None,
                 requirements: Optional[GraphRequirements] = None):
        self.requirements = requirements
        self.parameters = parameters
        self.log = default_log(self)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def update_requirements(self,
                            parameters: Optional['AlgorithmParameters'] = None,
                            requirements: Optional[GraphRequirements] = None):
        if requirements:
            self.requirements = requirements
        if parameters:
            self.parameters = parameters
