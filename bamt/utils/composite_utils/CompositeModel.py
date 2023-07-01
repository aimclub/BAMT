from typing import Optional, Union, List

from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph_node import LinkedGraphNode


class CompositeNode(LinkedGraphNode):
    def __str__(self):
        return self.content["name"]


class CompositeModel(GraphDelegate):
    def __init__(self,
                 nodes: Optional[Union[LinkedGraphNode,
                                       List[LinkedGraphNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1
