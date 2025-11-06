from .graph import Graph
from .dag import DirectedAcyclicGraph

DAG = DirectedAcyclicGraph  # Alias for convenience

__all__ = ['Graph', 'DirectedAcyclicGraph', 'DAG']
