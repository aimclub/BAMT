from dataclasses import dataclass

from golem.core.optimisers.optimization_parameters import GraphRequirements


@dataclass
class DynamicGraphRequirements(GraphRequirements):
    """ Class for using custom domain specific graph requirements. """
    def __init__(self, attributes: dict):
        for attribute, value in attributes.items():
            setattr(self, attribute, value)
