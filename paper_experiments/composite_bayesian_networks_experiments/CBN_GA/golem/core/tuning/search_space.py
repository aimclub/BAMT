from typing import Dict, Callable, List, Union

OperationParametersMapping = Dict[str, Dict[str, Dict[str, Union[Callable, List, str]]]]


class SearchSpace:
    """
    Args:
        search_space: dictionary with parameters and their search_space
            {'operation_name': {'param_name': {'hyperopt-dist': hyperopt distribution function,
                                               'sampling-scope': [sampling scope],
                                               'type': 'discrete', 'continuous' or 'categorical'}, ...}, ...},

            e.g. ``{'operation_name': {'param1': {'hyperopt-dist': hp.uniformint,
                                                  'sampling-scope': [2, 21]),
                                                  'type': 'discrete'},
                                       'param2': {'hyperopt-dist': hp.loguniform,
                                                  'sampling-scope': [0.001, 1]),
                                                  'type': 'continuous'},
                                       'param3': {'hyperopt-dist': hp.choice,
                                                  'sampling-scope': [['svd', 'lsqr', 'eigen']),
                                                  'type': 'categorical'}...}, ..}
    """

    def __init__(self, search_space: OperationParametersMapping):
        self.parameters_per_operation = search_space

    def get_parameters_for_operation(self, operation_name: str) -> List[str]:
        parameters_list = list(self.parameters_per_operation.get(operation_name, {}).keys())
        return parameters_list


def get_node_operation_parameter_label(node_id: int, operation_name: str, parameter_name: str) -> str:
    # Name with operation and parameter
    op_parameter_name = ''.join((operation_name, ' | ', parameter_name))

    # Name with node id || operation | parameter
    node_op_parameter_name = ''.join((str(node_id), ' || ', op_parameter_name))
    return node_op_parameter_name


def convert_parameters(parameters):
    """
    Function removes labels from dictionary with operations

    Args:
        parameters: labeled parameters

    Returns:
        new_parameters: dictionary without labels of node_id and operation_name
    """

    new_parameters = {}
    for operation_parameter, value in parameters.items():
        # Remove right part of the parameter name
        parameter_name = operation_parameter.split(' | ')[-1]

        if value is not None:
            new_parameters.update({parameter_name: value})

    return new_parameters
