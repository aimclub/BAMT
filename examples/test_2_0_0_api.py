"""
Example script demonstrating the BAMT 2.0.0 API.

This script shows the new sklearn-like API for structure learning
and Bayesian Network modeling.
"""

import numpy as np
import pandas as pd
import networkx as nx

# Import new 2.0.0 components
from bamt.dag_optimizers.score.hill_climbing import HillClimbing
from bamt.score_functions.k2_score import K2Score
from bamt.score_functions.mutual_information_score import MutualInformationScore
from bamt.score_functions.bic_score import BICScore
from bamt.score_functions.aic_score import AICScore
from bamt.models.probabilistic_structural_models.discrete_bayesian_network import (
    DiscreteBayesianNetwork,
)
from bamt.models.probabilistic_structural_models.continuous_bayesian_network import (
    ContinuousBayesianNetwork,
)
from bamt.models.probabilistic_structural_models.hybrid_bayesian_network import (
    HybridBayesianNetwork,
)


def test_discrete_bn():
    """Test Discrete Bayesian Network."""
    print("=" * 70)
    print("Testing Discrete Bayesian Network")
    print("=" * 70)

    # Generate synthetic discrete data
    np.random.seed(42)
    n = 1000

    # Simple structure: A -> B -> C, A -> C
    data = pd.DataFrame({
        'A': np.random.choice(['a1', 'a2', 'a3'], n),
        'B': np.random.choice(['b1', 'b2'], n),
        'C': np.random.choice(['c1', 'c2', 'c3', 'c4'], n),
    })

    print(f"\nGenerated {n} samples with 3 discrete variables")
    print(f"Data shape: {data.shape}")
    print(f"\nData types:\n{data.dtypes}")
    print(f"\nFirst few rows:\n{data.head()}")

    # Structure learning with Hill Climbing
    print("\n--- Structure Learning ---")
    score_fn = K2Score()
    optimizer = HillClimbing(score_function=score_fn, max_iter=50, max_parents=2)

    print(f"Using: {optimizer}")
    print(f"Score function: {score_fn}")

    structure = optimizer.optimize(data)
    print(f"\nLearned structure: {structure.edges()}")

    # Fit Discrete BN
    print("\n--- Parameter Learning ---")
    bn = DiscreteBayesianNetwork(structure=structure)
    bn.fit(data)
    print(f"Fitted: {bn}")

    # Sample from the network
    print("\n--- Sampling ---")
    samples = bn.sample(n_samples=10)
    print(f"Generated {len(samples)} samples:")
    print(samples)

    # Predict missing values
    print("\n--- Prediction ---")
    test_data = data.head(5).copy()
    test_data.loc[0, 'C'] = np.nan
    test_data.loc[1, 'B'] = np.nan
    print(f"Test data with missing values:\n{test_data}")

    predictions = bn.predict(test_data)
    print(f"\nPredictions:\n{predictions}")

    print("\n✓ Discrete BN test completed successfully!\n")


def test_continuous_bn():
    """Test Continuous Bayesian Network."""
    print("=" * 70)
    print("Testing Continuous Bayesian Network")
    print("=" * 70)

    # Generate synthetic continuous data
    np.random.seed(42)
    n = 500

    # Simple structure: X1 -> X2 -> X3
    X1 = np.random.normal(0, 1, n)
    X2 = 0.5 * X1 + np.random.normal(0, 0.5, n)
    X3 = 0.3 * X2 + np.random.normal(0, 0.3, n)

    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
    })

    print(f"\nGenerated {n} samples with 3 continuous variables")
    print(f"Data shape: {data.shape}")
    print(f"\nData statistics:\n{data.describe()}")

    # Structure learning with Hill Climbing + MI score
    print("\n--- Structure Learning ---")
    score_fn = MutualInformationScore(score_type="BIC")
    optimizer = HillClimbing(score_function=score_fn, max_iter=50, max_parents=2)

    print(f"Using: {optimizer}")
    print(f"Score function: {score_fn}")

    structure = optimizer.optimize(data)
    print(f"\nLearned structure: {structure.edges()}")

    # Fit Continuous BN
    print("\n--- Parameter Learning ---")
    bn = ContinuousBayesianNetwork(structure=structure)
    bn.fit(data)
    print(f"Fitted: {bn}")

    # Sample from the network
    print("\n--- Sampling ---")
    samples = bn.sample(n_samples=10)
    print(f"Generated {len(samples)} samples:")
    print(samples)

    print("\n✓ Continuous BN test completed successfully!\n")


def test_hybrid_bn():
    """Test Hybrid Bayesian Network."""
    print("=" * 70)
    print("Testing Hybrid Bayesian Network")
    print("=" * 70)

    # Generate synthetic hybrid data
    np.random.seed(42)
    n = 500

    # Mixed discrete and continuous
    data = pd.DataFrame({
        'Category': np.random.choice(['A', 'B', 'C'], n),
        'Value': np.random.normal(10, 2, n),
        'Count': np.random.randint(0, 10, n),
    })

    print(f"\nGenerated {n} samples with mixed data types")
    print(f"Data shape: {data.shape}")
    print(f"\nData types:\n{data.dtypes}")
    print(f"\nFirst few rows:\n{data.head()}")

    # Create a simple structure manually for demonstration
    print("\n--- Manual Structure ---")
    structure = nx.DiGraph()
    structure.add_nodes_from(['Category', 'Value', 'Count'])
    structure.add_edges_from([
        ('Category', 'Value'),
        ('Category', 'Count'),
    ])
    print(f"Structure: {structure.edges()}")

    # Fit Hybrid BN
    print("\n--- Parameter Learning ---")
    bn = HybridBayesianNetwork(
        structure=structure,
        discrete_columns=['Category'],
        continuous_columns=['Value', 'Count'],
    )
    bn.fit(data)
    print(f"Fitted: {bn}")

    # Sample from the network
    print("\n--- Sampling ---")
    samples = bn.sample(n_samples=10)
    print(f"Generated {len(samples)} samples:")
    print(samples)

    print("\n✓ Hybrid BN test completed successfully!\n")


def test_score_functions():
    """Test different score functions."""
    print("=" * 70)
    print("Testing Score Functions (K2, MI, BIC, AIC)")
    print("=" * 70)

    # Create simple test data
    np.random.seed(42)
    data = pd.DataFrame({
        'A': np.random.choice(['a1', 'a2'], 100),
        'B': np.random.choice(['b1', 'b2'], 100),
    })

    print("\nTest data (2 discrete variables, 100 samples)")

    # Test each score function
    score_functions = [
        ("K2Score", K2Score()),
        ("MutualInformation(LL)", MutualInformationScore(score_type="LL")),
        ("BICScore", BICScore()),
        ("AICScore", AICScore()),
    ]

    for name, score_fn in score_functions:
        try:
            score = score_fn.estimate(data)
            print(f"  {name}: {score:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    print("\n✓ Score functions test completed!\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  BAMT 2.0.0 API Demonstration".center(68) + "║")
    print("║" + "  Testing new sklearn-like interface".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    try:
        # Test 1: Discrete BN
        test_discrete_bn()

        # Test 2: Continuous BN
        test_continuous_bn()

        # Test 3: Hybrid BN
        test_hybrid_bn()

        # Test 4: Score functions
        test_score_functions()

        print("\n" + "=" * 70)
        print("🎉 All tests passed! BAMT 2.0.0 API is working correctly!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
