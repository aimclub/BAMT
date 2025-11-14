"""
Greedy Hill-Climbing optimizer for Bayesian Network structure learning.

This module implements a greedy search algorithm that explores the space of
possible DAG structures by testing three operations: add edge, delete edge,
and reverse edge.
"""

from typing import List, Optional, Tuple

import networkx as nx
import pandas as pd

from bamt.external.pyBN.utils.graph import would_cause_cycle
from bamt.score_functions import ScoreFunction

from .score_dag_optimizer import ScoreDAGOptimizer


class HillClimbing(ScoreDAGOptimizer):
    """
    Greedy Hill Climbing structure optimizer.

    Hill Climbing proceeds by choosing the move which maximizes the increase
    in fitness at the current step. It continues until no feasible single move
    increases the network fitness.

    The possible moves are:
        - Add edge
        - Delete edge
        - Reverse edge

    Attributes:
        score_function: The scoring function to optimize
        max_iter: Maximum number of iterations (default: 200)
        max_parents: Maximum number of parents per node (default: 3)
        init_edges: Initial edges to start with
        black_list: List of forbidden edges (u, v) tuples
        white_list: List of required/allowed edges (u, v) tuples
        debug: Whether to print debug information
    """

    def __init__(
        self,
        score_function: ScoreFunction,
        max_iter: int = 200,
        max_parents: int = 3,
        init_edges: Optional[List[Tuple[int, int]]] = None,
        black_list: Optional[List[Tuple[int, int]]] = None,
        white_list: Optional[List[Tuple[int, int]]] = None,
        debug: bool = False,
    ):
        """
        Initialize Hill Climbing optimizer.

        Args:
            score_function: Scoring function (K2Score, MutualInformationScore, etc.)
            max_iter: Maximum iterations
            max_parents: Maximum parents per node
            init_edges: Initial edge list
            black_list: Forbidden edges
            white_list: Allowed edges (for MMHC)
            debug: Print debug info
        """
        super().__init__()
        self.score_function = score_function
        self.max_iter = max_iter
        self.max_parents = max_parents
        self.init_edges = init_edges or []
        self.black_list = black_list or []
        self.white_list = white_list
        self.debug = debug
        self._cache = {}

    def optimize(
        self,
        data: pd.DataFrame,
        node_names: Optional[List[str]] = None,
        **kwargs
    ) -> nx.DiGraph:
        """
        Find the optimal DAG structure using Hill Climbing.

        Args:
            data: The dataset to learn structure from
            node_names: Names of nodes (defaults to column names)
            **kwargs: Additional parameters

        Returns:
            nx.DiGraph: The learned DAG structure
        """
        if node_names is None:
            node_names = list(data.columns)

        nrow, ncol = data.shape
        nodes = list(range(ncol))

        # Initialize parent and children dictionaries
        c_dict = {n: [] for n in nodes}
        p_dict = {n: [] for n in nodes}

        # Add initial edges
        for u, v in self.init_edges:
            c_dict[u].append(v)
            p_dict[v].append(u)

        # Convert data to numpy for faster access
        data_array = data.values

        # Main optimization loop
        _iter = 0
        improvement = True

        while improvement and _iter < self.max_iter:
            improvement = False
            max_delta = 0
            max_operation = None
            max_arc = None

            if self.debug:
                print(f"ITERATION: {_iter}")

            # TEST ARC ADDITIONS
            for u in nodes:
                for v in nodes:
                    if (
                        v not in c_dict[u]
                        and u != v
                        and not would_cause_cycle(c_dict, u, v)
                        and len(p_dict[v]) < self.max_parents
                    ):
                        # Check constraints
                        if not self._is_edge_allowed(u, v):
                            continue

                        # Calculate score improvement
                        old_cols = (v,) + tuple(p_dict[v])
                        new_cols = old_cols + (u,)

                        delta = self._score_delta(data_array, old_cols, new_cols, nrow)

                        if delta > max_delta:
                            if self.debug:
                                print(f"Improved Arc Addition: ({u}, {v}), Delta: {delta}")
                            max_delta = delta
                            max_operation = "add"
                            max_arc = (u, v)

            # TEST ARC DELETIONS
            for u in nodes:
                for v in c_dict[u]:
                    old_cols = (v,) + tuple(p_dict[v])
                    new_cols = tuple([i for i in old_cols if i != u])

                    delta = self._score_delta(data_array, old_cols, new_cols, nrow)

                    if delta > max_delta:
                        # Don't delete initial edges if specified
                        if (u, v) in self.init_edges:
                            continue

                        if self.debug:
                            print(f"Improved Arc Deletion: ({u}, {v}), Delta: {delta}")
                        max_delta = delta
                        max_operation = "delete"
                        max_arc = (u, v)

            # TEST ARC REVERSALS
            for u in nodes:
                for v in c_dict[u]:
                    # Check if reversal would cause cycle
                    if (
                        not would_cause_cycle(c_dict, v, u, reverse=(u, v))
                        and len(p_dict[u]) < self.max_parents
                    ):
                        # Check constraints for reversed edge
                        if not self._is_edge_allowed(v, u):
                            continue

                        # Score after reversing edge
                        # Remove u->v, add v->u
                        old_u_cols = (u,) + tuple(p_dict[u])
                        old_v_cols = (v,) + tuple(p_dict[v])

                        new_u_parents = tuple(p_dict[u]) + (v,)
                        new_v_parents = tuple([p for p in p_dict[v] if p != u])

                        new_u_cols = (u,) + new_u_parents
                        new_v_cols = (v,) + new_v_parents

                        delta_u = self._score_delta(data_array, old_u_cols, new_u_cols, nrow)
                        delta_v = self._score_delta(data_array, old_v_cols, new_v_cols, nrow)
                        delta = delta_u + delta_v

                        if delta > max_delta:
                            if self.debug:
                                print(f"Improved Arc Reversal: ({u}, {v}) -> ({v}, {u}), Delta: {delta}")
                            max_delta = delta
                            max_operation = "reverse"
                            max_arc = (u, v)

            # Apply best operation
            if max_delta > 0:
                improvement = True
                u, v = max_arc

                if max_operation == "add":
                    c_dict[u].append(v)
                    p_dict[v].append(u)
                elif max_operation == "delete":
                    c_dict[u].remove(v)
                    p_dict[v].remove(u)
                elif max_operation == "reverse":
                    c_dict[u].remove(v)
                    p_dict[v].remove(u)
                    c_dict[v].append(u)
                    p_dict[u].append(v)

                if self.debug:
                    print(f"Applied {max_operation}: {max_arc}, Delta: {max_delta}")

            _iter += 1

        # Convert to NetworkX DiGraph
        G = nx.DiGraph()
        G.add_nodes_from(node_names)

        for u in nodes:
            for v in c_dict[u]:
                G.add_edge(node_names[u], node_names[v])

        return G

    def _is_edge_allowed(self, u: int, v: int) -> bool:
        """Check if edge (u, v) is allowed by constraints."""
        # Check black list
        if (u, v) in self.black_list:
            return False

        # Check white list (if specified, only whitelisted edges allowed)
        if self.white_list is not None and (u, v) not in self.white_list:
            return False

        return True

    def _score_delta(
        self, data_array, old_cols: tuple, new_cols: tuple, nrow: int
    ) -> float:
        """
        Calculate the score improvement for changing parents.

        Args:
            data_array: Data as numpy array
            old_cols: Old column configuration (node + old parents)
            new_cols: New column configuration (node + new parents)
            nrow: Number of rows

        Returns:
            float: Score improvement (positive means improvement)
        """
        # Use cached scores if available
        if old_cols not in self._cache:
            old_data = data_array[:, old_cols]
            self._cache[old_cols] = self.score_function.estimate(
                pd.DataFrame(old_data)
            )

        if new_cols not in self._cache:
            new_data = data_array[:, new_cols]
            self._cache[new_cols] = self.score_function.estimate(
                pd.DataFrame(new_data)
            )

        # For MI-based scores, delta is typically: nrow * (mi_old - mi_new)
        # But scores are already computed, so we use: new_score - old_score
        # However, the sign convention may vary by score function
        # For compatibility with original HC implementation:
        delta = self._cache[new_cols] - self._cache[old_cols]

        return delta
