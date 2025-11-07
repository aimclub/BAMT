try:
    from .graph import Graph
    from .dag import DirectedAcyclicGraph

    DAG = DirectedAcyclicGraph  # Alias for convenience

    __all__ = ['Graph', 'DirectedAcyclicGraph', 'DAG']
except ImportError:
    # If networkx is not available, only export the names
    __all__ = []

