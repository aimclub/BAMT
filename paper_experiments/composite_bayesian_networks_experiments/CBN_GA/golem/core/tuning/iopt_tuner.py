from dataclasses import dataclass, field
from typing import List, Dict, Generic, Tuple, Any, Optional

import numpy as np
from iOpt.method.listener import ConsoleFullOutputListener
from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate
from golem.core.tuning.search_space import SearchSpace, get_node_operation_parameter_label
from golem.core.tuning.tuner_interface import BaseTuner, DomainGraphForTune


@dataclass
class IOptProblemParameters:
    float_parameters_names: List[str] = field(default_factory=list)
    discrete_parameters_names: List[str] = field(default_factory=list)
    lower_bounds_of_float_parameters: List[float] = field(default_factory=list)
    upper_bounds_of_float_parameters: List[float] = field(default_factory=list)
    discrete_parameters_vals: List[List[Any]] = field(default_factory=list)

    @staticmethod
    def from_parameters_dicts(float_parameters_dict: Optional[Dict[str, List]] = None,
                              discrete_parameters_dict: Optional[Dict[str, List]] = None):
        float_parameters_dict = float_parameters_dict or {}
        discrete_parameters_dict = discrete_parameters_dict or {}

        float_parameters_names = list(float_parameters_dict.keys())
        discrete_parameters_names = list(discrete_parameters_dict.keys())

        lower_bounds_of_float_parameters = [bounds[0] for bounds in float_parameters_dict.values()]
        upper_bounds_of_float_parameters = [bounds[1] for bounds in float_parameters_dict.values()]
        discrete_parameters_vals = [values_set for values_set in discrete_parameters_dict.values()]

        # TODO: Remove - for now IOpt handles only float variables, so we treat discrete parameters as float ones
        float_parameters_names.extend(discrete_parameters_names)
        lower_bounds_of_discrete_parameters = [bounds[0] for bounds in discrete_parameters_dict.values()]
        upper_bounds_of_discrete_parameters = [bounds[1] for bounds in discrete_parameters_dict.values()]
        lower_bounds_of_float_parameters.extend(lower_bounds_of_discrete_parameters)
        upper_bounds_of_float_parameters.extend(upper_bounds_of_discrete_parameters)

        return IOptProblemParameters(float_parameters_names, discrete_parameters_names,
                                     lower_bounds_of_float_parameters,
                                     upper_bounds_of_float_parameters, discrete_parameters_vals)


class GolemProblem(Problem, Generic[DomainGraphForTune]):
    def __init__(self, graph: DomainGraphForTune,
                 objective_evaluate: ObjectiveEvaluate,
                 problem_parameters: IOptProblemParameters):
        super().__init__()
        self.objective_evaluate = objective_evaluate
        self.graph = graph

        self.numberOfObjectives = 1
        self.numberOfConstraints = 0

        self.discreteVariableNames = problem_parameters.discrete_parameters_names
        self.discreteVariableValues = problem_parameters.discrete_parameters_vals
        self.numberOfDiscreteVariables = len(self.discreteVariableNames)

        self.floatVariableNames = problem_parameters.float_parameters_names
        self.lowerBoundOfFloatVariables = problem_parameters.lower_bounds_of_float_parameters
        self.upperBoundOfFloatVariables = problem_parameters.upper_bounds_of_float_parameters
        self.numberOfFloatVariables = len(self.floatVariableNames)

        self._default_metric_value = np.inf

    def Calculate(self, point: Point, functionValue: FunctionValue) -> FunctionValue:
        new_parameters = self.get_parameters_dict_from_iopt_point(point)
        BaseTuner.set_arg_graph(self.graph, new_parameters)
        graph_fitness = self.objective_evaluate(self.graph)
        metric_value = graph_fitness.value if graph_fitness.valid else self._default_metric_value
        functionValue.value = metric_value
        return functionValue

    def get_parameters_dict_from_iopt_point(self, point: Point) -> Dict[str, Any]:
        """Constructs a dict with all hyperparameters """
        float_parameters = dict(zip(self.floatVariableNames, point.floatVariables)) \
            if point.floatVariables is not None else {}
        discrete_parameters = dict(zip(self.discreteVariableNames, point.discreteVariables)) \
            if point.discreteVariables is not None else {}

        # TODO: Remove workaround - for now IOpt handles only float variables, so discrete parameters
        #  are optimized as continuous and we need to round them
        for parameter_name in float_parameters:
            if parameter_name in self.discreteVariableNames:
                float_parameters[parameter_name] = round(float_parameters[parameter_name])

        parameters_dict = {**float_parameters, **discrete_parameters}
        return parameters_dict


class IOptTuner(BaseTuner):
    """
    Base class for hyperparameters optimization based on hyperopt library

    Args:
        objective_evaluate: objective to optimize
        adapter: the function for processing of external object that should be optimized
        iterations: max number of iterations
        search_space: SearchSpace instance
        n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's)
        eps: The accuracy of the solution of the problem. Less value - higher search accuracy, less likely to stop
            prematurely.
        r: Reliability parameter. Higher r is slower convergence, higher probability of finding a global minimum.
        evolvent_density: Density of the evolvent. By default :math:`2^{-10}` on hypercube :math:`[0,1]^N`,
             which means, that the maximum search accuracy is :math:`2^{-10}`.
        eps_r: Parameter that affects the speed of solving the task. epsR = 0 - slow convergence
             to the exact solution, epsR>0 - quick converge to the neighborhood of the solution.
        refine_solution: if true, then the solution will be refined with local search.
        deviation: required improvement (in percent) of a metric to return tuned graph.
            By default, ``deviation=0.05``, which means that tuned graph will be returned
            if it's metric will be at least 0.05% better than the initial.
    """

    def __init__(self, objective_evaluate: ObjectiveEvaluate,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 n_jobs: int = -1,
                 eps: float = 0.01,
                 r: float = 2.0,
                 evolvent_density: int = 10,
                 eps_r: float = 0.001,
                 refine_solution: bool = False,
                 deviation: float = 0.05, **kwargs):
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations=iterations,
                         n_jobs=n_jobs,
                         deviation=deviation, **kwargs)
        self.solver_parameters = SolverParameters(r=np.double(r),
                                                  eps=np.double(eps),
                                                  itersLimit=iterations,
                                                  evolventDensity=evolvent_density,
                                                  epsR=np.double(eps_r),
                                                  refineSolution=refine_solution)

    def tune(self, graph: DomainGraphForTune, show_progress: bool = True) -> DomainGraphForTune:
        graph = self.adapter.adapt(graph)
        problem_parameters, initial_parameters = self._get_parameters_for_tune(graph)

        no_parameters_to_optimize = (not problem_parameters.discrete_parameters_names and
                                     not problem_parameters.float_parameters_names)
        self.init_check(graph)

        if no_parameters_to_optimize:
            self._stop_tuning_with_message(f'Graph "{graph.graph_description}" has no parameters to optimize')
            final_graph = graph
        else:
            if initial_parameters:
                initial_point = Point(**initial_parameters)
                self.solver_parameters.startPoint = initial_point

            problem = GolemProblem(graph, self.objective_evaluate, problem_parameters)
            solver = Solver(problem, parameters=self.solver_parameters)

            if show_progress:
                console_output = ConsoleFullOutputListener(mode='full')
                solver.AddListener(console_output)

            solution = solver.Solve()
            best_point = solution.bestTrials[0].point
            best_parameters = problem.get_parameters_dict_from_iopt_point(best_point)
            final_graph = self.set_arg_graph(graph, best_parameters)

            self.was_tuned = True

        # Validate if optimisation did well
        graph = self.final_check(final_graph)
        final_graph = self.adapter.restore(graph)
        return final_graph

    def _get_parameters_for_tune(self, graph: OptGraph) -> Tuple[IOptProblemParameters, dict]:
        """ Method for defining the search space

        Args:
            graph: graph to be tuned

        Returns:
            parameters_dict: dict with operation names and parameters
            initial_parameters: dict with initial parameters of the graph
        """
        float_parameters_dict = {}
        discrete_parameters_dict = {}
        initial_parameters = {'floatVariables': [], 'discreteVariables': []}
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the graph
            float_node_parameters, discrete_node_parameters = get_node_parameters_for_iopt(self.search_space,
                                                                                           node_id,
                                                                                           operation_name)

            # Set initial parameters for search
            for parameter, bounds in float_node_parameters.items():
                # If parameter is not set use parameter minimum possible value
                initaial_value = node.parameters.get(parameter) or bounds[0]
                initial_parameters['floatVariables'].append(initaial_value)

            for parameter, bounds in discrete_node_parameters.items():
                # If parameter is not set use parameter minimum possible value
                initaial_value = node.parameters.get(parameter) or bounds[0]
                initial_parameters['discreteVariables'].append(initaial_value)

            float_parameters_dict.update(float_node_parameters)
            discrete_parameters_dict.update(discrete_node_parameters)
        parameters_dict = IOptProblemParameters.from_parameters_dicts(float_parameters_dict, discrete_parameters_dict)
        return parameters_dict, initial_parameters


def get_node_parameters_for_iopt(search_space: SearchSpace, node_id: int, operation_name: str) \
        -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Method for forming dictionary with hyperparameters of node operation for the ``IOptTuner``

    Args:
        search_space: SearchSpace with parameters per operation
        node_id: number of node in graph.nodes list
        operation_name: name of operation in the node

    Returns:
        float_parameters_dict: dictionary-like structure with labeled float hyperparameters
        and their range per operation
        discrete_parameters_dict: dictionary-like structure with labeled discrete hyperparameters
        and their range per operation
    """
    # Get available parameters for operation
    parameters_dict = search_space.parameters_per_operation.get(operation_name, {})

    discrete_parameters_dict = {}
    float_parameters_dict = {}

    for parameter_name, parameter_properties in parameters_dict.items():
        node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

        parameter_type = parameter_properties.get('type')
        if parameter_type == 'discrete':
            discrete_parameters_dict.update({node_op_parameter_name: parameter_properties
                                            .get('sampling-scope')})
        elif parameter_type == 'continuous':
            float_parameters_dict.update({node_op_parameter_name: parameter_properties
                                         .get('sampling-scope')})

    return float_parameters_dict, discrete_parameters_dict
