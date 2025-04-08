from typing import Dict, Type, List, Any, Optional
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier

from bamt.builders.evo_builders.deap_graph import Graph, Node


class MLModels:
    """Utility class for managing machine learning models."""

    def __init__(self):
        # Dictionary mapping model names to model classes
        self.dict_models = {
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression,
            "RandomForestRegressor": RandomForestRegressor,
            "RandomForestClassifier": RandomForestClassifier,
            "SVR": SVR,
            "SVC": SVC,
            "MLPRegressor": MLPRegressor,
            "MLPClassifier": MLPClassifier,
        }

    def get_model_by_name(self, model_name: str):
        """Get a model class by its name."""
        if model_name in self.dict_models:
            return self.dict_models[model_name]()
        return None

    def get_model_by_children_type(self, node: Node) -> Optional[Any]:
        """Get an appropriate model based on node type and parents."""
        if "type" not in node.content:
            return None

        node_type = node.content["type"]

        # Count discrete and continuous parents
        disc_parents = [
            p
            for p in node.parents
            if "type" in p.content and p.content["type"] in ["disc", "disc_num"]
        ]
        cont_parents = [
            p
            for p in node.parents
            if "type" in p.content and p.content["type"] == "cont"
        ]

        # No parents, return None
        if not disc_parents and not cont_parents:
            return None

        # Select appropriate model based on node type and parent types
        if node_type == "cont":
            # For continuous target
            if len(disc_parents) > 0 and len(cont_parents) > 0:
                # Mixed parents
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif len(cont_parents) > 0:
                # Only continuous parents
                return LinearRegression()
            else:
                # Only discrete parents
                return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            # For discrete target
            if len(disc_parents) > 0 and len(cont_parents) > 0:
                # Mixed parents
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif len(cont_parents) > 0:
                # Only continuous parents
                return LogisticRegression(max_iter=1000)
            else:
                # Only discrete parents
                return RandomForestClassifier(n_estimators=100, random_state=42)
