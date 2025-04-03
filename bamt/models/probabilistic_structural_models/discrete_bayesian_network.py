from bamt.models.probabilistic_structural_models.bayesian_network import BayesianNetwork
from bamt.checkers.network_checker import NetworkChecker
from bamt.local_typing.type_manager import TypeManager
from bamt.core.graph.dag import DirectedAcyclicGraph
from bamt.local_typing.node_types import NodeType
from bamt.core.nodes.root_nodes import ContinuousNode, DiscreteNode
from bamt.core.nodes.child_nodes import ConditionalDiscreteNode, ConditionalContinuousNode
from bamt.loggers.logger import logger_bn


class DiscreteBayesianNetwork(BayesianNetwork):
    def __init__(self):
        super().__init__()
        self.checker = None
        self.nodes = []
        self.edges = []

    def validate_graph(self, graph):
        return True

    def get_node2type(self, data, graph: DirectedAcyclicGraph = None):
        # todo
        type_manager = TypeManager()
        descriptor = type_manager.get_descriptor(data)
        self.checker = NetworkChecker(descriptor)
        # fit means here only parameters learning,
        # but before this we need get info about data and network (e.g., typing for net)

        if self.validate_graph(graph):
            family = graph.get_family(descriptor)
            node2type = type_manager.find_node_types(family, descriptor)
        else:
            logger_bn.error('Graph validation failed.')

        return node2type

    @staticmethod
    def get_node_instance(name, node_type):
        match node_type:
            case NodeType.root_discrete:
                return DiscreteNode(name)
            case NodeType.root_continuous:
                return ContinuousNode(name)
            case NodeType.conditional_discrete:
                return ConditionalDiscreteNode(name)
            case NodeType.conditional_continuous:
                return ConditionalContinuousNode(name)
            case _:
                logger_bn.error(f"Unknown node type {node_type}")

    def from_dag(self, data, graph):
        node2type = self.get_node2type(data, graph)
        # self.nodes = [self.get_node_instance(name, node2type[name]) for name in data.columns]
        for name in data.columns:
            self.nodes.append(self.get_node_instance(name, node2type[name]))
        return self

    def fit(self, data, parameters_estimator=None):
        pass

    def from_digraph(self):
        pass

    def predict(self, data):
        pass

    def sample(self):
        pass

    def __str__(self):
        return "Discrete Bayesian Network"
