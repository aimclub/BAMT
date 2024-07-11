from golem.core.dag.linked_graph_node import LinkedGraphNode

class BNNode(LinkedGraphNode):
    def __str__(self):
        return self.content["name"]