
from golem.core.optimisers.graph import OptNode
from golem.core.dag.linked_graph_node import LinkedGraphNode

class CompositeNode(LinkedGraphNode):
    def __str__(self):
        return self.content["name"]