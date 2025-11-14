"""
Example script demonstrating CompositeBayesianNetwork usage.

CompositeBayesianNetwork provides explicit control over which ML models
are used for each conditional node, allowing you to use XGBoost, CatBoost,
LightGBM, or any sklearn-compatible model.
"""

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from bamt.dag_optimizers.score.hill_climbing import HillClimbing
from bamt.models.probabilistic_structural_models.composite_bayesian_network import (
    CompositeBayesianNetwork,
)
from bamt.score_functions.k2_score import K2Score

# Try to import optional advanced models
try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def test_composite_bn_with_custom_models():
    """Test CompositeBN with custom ML models."""
    print("\n" + "=" * 70)
    print("Test 1: CompositeBN with Custom ML Models")
    print("=" * 70)

    # Create synthetic mixed data
    np.random.seed(42)
    n = 500

    # Generate data with known structure: A -> B -> C, D -> C
    A = np.random.choice(["low", "medium", "high"], size=n, p=[0.3, 0.5, 0.2])
    A_encoded = pd.Categorical(A).codes

    B = A_encoded * 2.5 + np.random.normal(0, 1, n)

    C_base = B * 1.5 + A_encoded * 0.5
    C = (C_base > np.median(C_base)).astype(int)

    D = np.random.normal(10, 2, n)

    data = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})

    print(f"\nDataset shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"\nFirst few rows:\n{data.head()}")

    # Define structure
    structure = nx.DiGraph()
    structure.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"), ("D", "C")])

    # Create CompositeBN with explicit column types
    bn = CompositeBayesianNetwork(
        structure=structure, discrete_columns=["A", "C"], continuous_columns=["B", "D"]
    )

    # Set custom models for specific nodes
    # Node C is discrete with parents, so it uses a classifier
    bn.set_classifiers(
        {
            "C": RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42
            )
        }
    )

    # Node B is continuous with parent A, so it uses a regressor
    bn.set_regressors(
        {"B": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)}
    )

    print("\n2. Fitting CompositeBN...")
    bn.fit(data)
    print(f"   {bn}")

    # Test sampling
    print("\n3. Sampling from the network...")
    samples = bn.sample(n_samples=10)
    print(f"   Generated {len(samples)} samples")
    print(f"   Sample data:\n{samples.head()}")

    # Test prediction
    print("\n4. Testing prediction...")
    test_data = pd.DataFrame({"A": ["high", "low"], "D": [12.0, 8.0]})
    predictions = bn.predict(test_data, target_columns=["B", "C"])
    print(f"   Predictions:\n{predictions}")

    print("\n" + "=" * 70)


def test_composite_bn_with_advanced_models():
    """Test CompositeBN with XGBoost/CatBoost if available."""
    if not (HAS_XGB or HAS_CATBOOST):
        print("\n" + "=" * 70)
        print("Test 2: SKIPPED - XGBoost/CatBoost not available")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Test 2: CompositeBN with XGBoost/CatBoost")
    print("=" * 70)

    # Create data
    np.random.seed(123)
    n = 400

    X1 = np.random.choice([0, 1], size=n)
    X2 = X1 * 3 + np.random.normal(0, 0.5, n)
    Y = (X1 + X2 > 3).astype(int)

    data = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})

    print(f"\nDataset shape: {data.shape}")

    # Define structure
    structure = nx.DiGraph()
    structure.add_edges_from([("X1", "X2"), ("X1", "Y"), ("X2", "Y")])

    bn = CompositeBayesianNetwork(
        structure=structure, discrete_columns=["X1", "Y"], continuous_columns=["X2"]
    )

    # Set advanced models
    if HAS_XGB:
        print("\nUsing XGBoost for node Y")
        bn.set_classifiers(
            {
                "Y": XGBClassifier(
                    n_estimators=50, max_depth=3, random_state=42, verbosity=0
                )
            }
        )
        bn.set_regressors(
            {
                "X2": XGBRegressor(
                    n_estimators=50, max_depth=3, random_state=42, verbosity=0
                )
            }
        )
    elif HAS_CATBOOST:
        print("\nUsing CatBoost for node Y")
        bn.set_classifiers(
            {
                "Y": CatBoostClassifier(
                    iterations=50, depth=3, random_state=42, verbose=False
                )
            }
        )
        bn.set_regressors(
            {
                "X2": CatBoostRegressor(
                    iterations=50, depth=3, random_state=42, verbose=False
                )
            }
        )

    bn.fit(data)
    print(f"\n{bn}")

    # Sample
    samples = bn.sample(n_samples=5)
    print(f"\nGenerated samples:\n{samples}")

    print("\n" + "=" * 70)


def test_composite_bn_auto_selection():
    """Test CompositeBN with automatic model selection."""
    print("\n" + "=" * 70)
    print("Test 3: CompositeBN with Automatic Model Selection")
    print("=" * 70)

    # Create simple data
    np.random.seed(456)
    n = 300

    A = np.random.randint(0, 3, n)
    B = A + np.random.normal(0, 0.5, n)
    C = (B > 1).astype(int)

    data = pd.DataFrame({"A": A, "B": B, "C": C})

    print(f"\nDataset shape: {data.shape}")

    # Learn structure with Hill Climbing
    print("\nLearning structure with Hill Climbing...")
    score_fn = K2Score()
    optimizer = HillClimbing(score_function=score_fn, max_iter=50)
    structure = optimizer.optimize(data)

    print(f"Learned structure edges: {list(structure.edges())}")

    # Use CompositeBN with automatic model selection (no custom models)
    bn = CompositeBayesianNetwork(
        structure=structure, discrete_columns=["A", "C"], continuous_columns=["B"]
    )

    print("\nFitting with automatic model selection...")
    bn.fit(data)
    print(f"{bn}")

    # Sample and predict
    samples = bn.sample(n_samples=5)
    print(f"\nSamples:\n{samples}")

    test = pd.DataFrame({"A": [0, 2]})
    preds = bn.predict(test)
    print(f"\nPredictions:\n{preds}")

    print("\n" + "=" * 70)


def test_composite_bn_model_comparison():
    """Test CompositeBN with multiple candidate models."""
    print("\n" + "=" * 70)
    print("Test 4: CompositeBN with Multiple Candidate Models")
    print("=" * 70)

    # Create data
    np.random.seed(789)
    n = 350

    X = np.random.choice(["A", "B", "C"], size=n)
    X_enc = pd.Categorical(X).codes
    Y = X_enc * 2 + np.random.normal(0, 1, n)
    Z = (Y > 2).astype(int)

    data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    print(f"\nDataset shape: {data.shape}")

    structure = nx.DiGraph()
    structure.add_edges_from([("X", "Y"), ("Y", "Z")])

    bn = CompositeBayesianNetwork(
        structure=structure, discrete_columns=["X", "Z"], continuous_columns=["Y"]
    )

    # Provide multiple candidate models for node Z
    # The Classifier will automatically select the best one via CV
    bn.set_classifiers(
        {
            "Z": {
                "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
                "RandomForest": RandomForestClassifier(
                    n_estimators=30, max_depth=3, random_state=42
                ),
            }
        }
    )

    # Multiple regressors for node Y
    bn.set_regressors(
        {
            "Y": {
                "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42),
                "RandomForest": RandomForestRegressor(
                    n_estimators=30, max_depth=3, random_state=42
                ),
            }
        }
    )

    print("\nFitting with automatic model selection from candidates...")
    bn.fit(data)
    print(f"{bn}")

    samples = bn.sample(n_samples=5)
    print(f"\nSamples:\n{samples}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BAMT 2.0.0 - CompositeBayesianNetwork Examples")
    print("=" * 70)

    test_composite_bn_with_custom_models()
    test_composite_bn_with_advanced_models()
    test_composite_bn_auto_selection()
    test_composite_bn_model_comparison()

    print("\n" + "=" * 70)
    print("All CompositeBayesianNetwork tests completed!")
    print("=" * 70)
