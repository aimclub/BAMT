from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Dict, Generic, Tuple, Any, Optional

import numpy as np
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.problem import Problem
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.trial import Point, FunctionValue

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.optimisers.genetic.evaluation import determine_n_jobs
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate
from golem.core.tuning.search_space import SearchSpace, get_node_operation_parameter_label, convert_parameters
from golem.core.tuning.tuner_interface import BaseTuner, DomainGraphForTune
from golem.utilities.data_structures import ensure_wrapped_in_sequence


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

        return IOptProblemParameters(float_parameters_names,
                                     discrete_parameters_names,
                                     lower_bounds_of_float_parameters,
                                     upper_bounds_of_float_parameters,
                                     discrete_parameters_vals)


class GolemProblem(Problem, Generic[DomainGraphForTune]):
    def __init__(self, graph: DomainGraphForTune,
                 objective_evaluate: ObjectiveEvaluate,
                 problem_parameters: IOptProblemParameters,
                 objectives_number: int = 1):
        super().__init__()
        self.objective_evaluate = objective_evaluate
        self.graph = graph

        self.number_of_objectives = objectives_number
        self.number_of_constraints = 0

        self.discrete_variable_names = problem_parameters.discrete_parameters_names
        self.discrete_variable_values = problem_parameters.discrete_parameters_vals
        self.number_of_discrete_variables = len(self.discrete_variable_names)

        self.float_variable_names = problem_parameters.float_parameters_names
        self.lower_bound_of_float_variables = problem_parameters.lower_bounds_of_float_parameters
        self.upper_bound_of_float_variables = problem_parameters.upper_bounds_of_float_parameters
        self.number_of_float_variables = len(self.float_variable_names)

        self._default_metric_value = np.inf

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        new_parameters = self.get_parameters_dict_from_iopt_point(point)
        BaseTuner.set_arg_graph(self.graph, new_parameters)
        graph_fitness = self.objective_evaluate(self.graph)
        metric_value = graph_fitness.value if graph_fitness.valid else self._default_metric_value
        function_value.value = metric_value
        return function_value

    def get_parameters_dict_from_iopt_point(self, point: Point) -> Dict[str, Any]:
        """Constructs a dict with all hyperparameters """
        float_parameters = dict(zip(self.float_variable_names, point.float_variables)) \
            if point.float_variables is not None else {}
        discrete_parameters = dict(zip(self.discrete_variable_names, point.discrete_variables)) \
            if point.discrete_variables is not None else {}

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
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 eps: float = 0.001,
                 r: float = 2.0,
                 evolvent_density: int = 10,
                 eps_r: float = 0.001,
                 refine_solution: bool = False,
                 deviation: float = 0.05, **kwargs):
        super().__init__(objective_evaluate,
                         search_space,
                         adapter,
                         iterations=iterations,
                         timeout=timeout,
                         n_jobs=n_jobs,
                         deviation=deviation, **kwargs)
        self.n_jobs = determine_n_jobs(self.n_jobs)
        self.solver_parameters = SolverParameters(r=np.double(r),
                                                  eps=np.double(eps),
                                                  iters_limit=iterations,
                                                  evolvent_density=evolvent_density,
                                                  eps_r=np.double(eps_r),
                                                  refine_solution=refine_solution,
                                                  number_of_parallel_points=self.n_jobs,
                                                  timeout=round(timeout.total_seconds()/60) if self.timeout else -1)

    def _tune(self, graph: DomainGraphForTune, show_progress: bool = True) -> DomainGraphForTune:
        problem_parameters, initial_parameters = self._get_parameters_for_tune(graph)

        has_parameters_to_optimize = (len(problem_parameters.discrete_parameters_names) > 0 or
                                      len(problem_parameters.float_parameters_names) > 0)
        self.objectives_number = len(ensure_wrapped_in_sequence(self.init_metric))
        is_multi_objective = self.objectives_number > 1

        if self._check_if_tuning_possible(graph, has_parameters_to_optimize, supports_multi_objective=True):
            if initial_parameters:
                initial_point = Point(**initial_parameters)
                self.solver_parameters.start_point = initial_point

            problem = GolemProblem(graph, self.objective_evaluate, problem_parameters, self.objectives_number)
            solver = Solver(problem, parameters=self.solver_parameters)

            if show_progress:
                console_output = ConsoleOutputListener(mode='full')
                solver.add_listener(console_output)

            solver.solve()
            solution = solver.get_results()
            if not is_multi_objective:
                best_point = solution.best_trials[0].point
                best_parameters = problem.get_parameters_dict_from_iopt_point(best_point)
                tuned_graphs = self.set_arg_graph(graph, best_parameters)
                self.was_tuned = True
            else:
                tuned_graphs = []
                for best_trial in solution.best_trials:
                    best_parameters = problem.get_parameters_dict_from_iopt_point(best_trial.point)
                    tuned_graph = self.set_arg_graph(deepcopy(graph), best_parameters)
                    tuned_graphs.append(tuned_graph)
                    self.was_tuned = True
        else:
            tuned_graphs = graph

        return tuned_graphs

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
        has_init_parameters = any(len(node.parameters) > 0 for node in graph.nodes)
        initial_parameters = {'float_variables': [], 'discrete_variables': []} if has_init_parameters else None
        for node_id, node in enumerate(graph.nodes):
            operation_name = node.name

            # Assign unique prefix for each model hyperparameter
            # label - number of node in the graph
            float_node_parameters, discrete_node_parameters = get_node_parameters_for_iopt(
                self.search_space,
                node_id,
                operation_name)
            if has_init_parameters:
                # Set initial parameters for search
                for parameter, bounds in convert_parameters(float_node_parameters).items():
                    # If parameter is not set use parameter minimum possible value
                    initial_value = node.parameters.get(parameter) or bounds[0]
                    initial_parameters['float_variables'].append(initial_value)

                for parameter, values in convert_parameters(discrete_node_parameters).items():
                    # If parameter is not set use the last value
                    initial_value = node.parameters.get(parameter) or values[-1]
                    initial_parameters['discrete_variables'].append(initial_value)

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
    categorical_parameters_dict = {}

    for parameter_name, parameter_properties in parameters_dict.items():
        node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

        parameter_type = parameter_properties.get('type')
        if parameter_type == 'discrete':
            discrete_parameters_dict.update({node_op_parameter_name: list(range(*parameter_properties
                                                                                .get('sampling-scope')))})
        elif parameter_type == 'continuous':
            float_parameters_dict.update({node_op_parameter_name: parameter_properties
                                         .get('sampling-scope')})
        elif parameter_type == 'categorical':
            categorical_parameters_dict.update({node_op_parameter_name: parameter_properties
                                               .get('sampling-scope')[0]})

    # IOpt does not distinguish between discrete and categorical parameters
    discrete_parameters_dict = {**discrete_parameters_dict, **categorical_parameters_dict}
    return float_parameters_dict, discrete_parameters_dict
