"""
Example demonstrating new BAMT 2.0.0 features:
- Serialization (save/load)
- Visualization
- Preprocessing utilities
- PC Algorithm

This script shows the latest features added to the 2.0.0 architecture.
"""

import numpy as np
import pandas as pd
import tempfile
import os

# Import new utilities
from bamt.utils.preprocessing import DataPreprocessor

# Import optimizers
from bamt.dag_optimizers.score.hill_climbing import HillClimbing
from bamt.dag_optimizers.constraint.pc_algorithm import PCAlgorithm
from bamt.score_functions.k2_score import K2Score

# Import networks
from bamt.models.probabilistic_structural_models.discrete_bayesian_network import (
    DiscreteBayesianNetwork,
)


def test_preprocessing():
    """Test preprocessing utilities."""
    print("=" * 70)
    print("Testing Preprocessing Utilities")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n = 500

    data = pd.DataFrame({
        "Discrete1": np.random.choice(["A", "B", "C"], n),
        "Discrete2": np.random.choice(["X", "Y"], n),
        "Continuous1": np.random.normal(10, 2, n),
        "Continuous2": np.random.normal(5, 1, n),
    })

    print(f"\nOriginal data shape: {data.shape}")
    print(f"\nFirst rows:\n{data.head()}")

    # 1. Infer column types
    print("\n--- Column Type Inference ---")
    column_types = DataPreprocessor.infer_column_types(data)
    print(f"Inferred types: {column_types}")

    # 2. Encode discrete columns
    print("\n--- Discrete Column Encoding ---")
    data_encoded, encoding = DataPreprocessor.encode_discrete_columns(data)
    print(f"Encoded data:\n{data_encoded.head()}")
    print(f"Encoding mappings: {encoding}")

    # 3. Split columns by type
    print("\n--- Column Splitting ---")
    disc_cols, cont_cols = DataPreprocessor.split_columns_by_type(data)
    print(f"Discrete columns: {disc_cols}")
    print(f"Continuous columns: {cont_cols}")

    # 4. Discretize continuous columns
    print("\n--- Continuous Column Discretization ---")
    data_all_discrete, discretizers = DataPreprocessor.discretize_continuous_columns(
        data_encoded, continuous_columns=cont_cols, n_bins=3
    )
    print(f"Fully discretized data:\n{data_all_discrete.head()}")

    print("\n✓ Preprocessing test completed!\n")
    return data_all_discrete


def test_serialization_and_pc():
    """Test serialization and PC algorithm."""
    print("=" * 70)
    print("Testing Serialization & PC Algorithm")
    print("=" * 70)

    # Generate discrete data
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "A": np.random.choice(["a1", "a2", "a3"], n),
        "B": np.random.choice(["b1", "b2"], n),
        "C": np.random.choice(["c1", "c2", "c3"], n),
        "D": np.random.choice(["d1", "d2"], n),
    })

    print(f"\nData shape: {data.shape}")

    # Test PC Algorithm
    print("\n--- PC Algorithm Structure Learning ---")
    try:
        pc_optimizer = PCAlgorithm(alpha=0.05)
        print(f"Using: {pc_optimizer}")

        structure = pc_optimizer.optimize(data)
        print(f"Learned structure (PC): {list(structure.edges())}")

        # Fit BN
        bn = DiscreteBayesianNetwork(structure=structure)
        bn.fit(data)
        print(f"Fitted: {bn}")

    except ImportError as e:
        print(f"PC Algorithm requires pgmpy. Skipping PC test. Error: {e}")
        # Fall back to Hill Climbing
        print("\nFalling back to Hill Climbing...")
        score_fn = K2Score()
        hc_optimizer = HillClimbing(score_function=score_fn, max_iter=50)
        structure = hc_optimizer.optimize(data)
        print(f"Learned structure (HC): {list(structure.edges())}")

        bn = DiscreteBayesianNetwork(structure=structure)
        bn.fit(data)
        print(f"Fitted: {bn}")

    # Test Serialization
    print("\n--- Serialization (Save/Load) ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model")

        # Save
        bn.save(model_path)
        print(f"✓ Model saved to {model_path}.json and {model_path}.pkl")

        # Load
        bn_loaded = DiscreteBayesianNetwork.load(model_path)
        print(f"✓ Model loaded: {bn_loaded}")

        # Verify structure is the same
        original_edges = set(bn.structure.edges())
        loaded_edges = set(bn_loaded.structure.edges())
        assert original_edges == loaded_edges, "Loaded structure differs!"
        print("✓ Loaded structure matches original")

        # Test sampling with loaded model
        samples = bn_loaded.sample(n_samples=5)
        print(f"\nSamples from loaded model:\n{samples}")

    print("\n✓ Serialization test completed!\n")
    return bn


def test_visualization(bn):
    """Test visualization features."""
    print("=" * 70)
    print("Testing Visualization")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test interactive visualization
        print("\n--- Interactive Visualization (HTML) ---")
        html_path = os.path.join(tmpdir, "network_interactive.html")
        try:
            bn.plot(html_path, mode="interactive")
            print(f"✓ Interactive plot saved to {html_path}")
            print(f"  File size: {os.path.getsize(html_path)} bytes")
        except ImportError as e:
            print(f"⚠ Visualization requires pyvis. Error: {e}")

        # Test static visualization
        print("\n--- Static Visualization (PNG) ---")
        png_path = os.path.join(tmpdir, "network_static.png")
        try:
            bn.plot(png_path, mode="static", figsize=(10, 6))
            print(f"✓ Static plot saved to {png_path}")
            print(f"  File size: {os.path.getsize(png_path)} bytes")
        except Exception as e:
            print(f"⚠ Static visualization failed. Error: {e}")

    print("\n✓ Visualization test completed!\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  BAMT 2.0.0 - New Features Demonstration".center(68) + "║")
    print("║" + "  Serialization, Visualization, Preprocessing, PC".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    try:
        # Test 1: Preprocessing
        preprocessed_data = test_preprocessing()

        # Test 2: Serialization & PC Algorithm
        bn = test_serialization_and_pc()

        # Test 3: Visualization
        test_visualization(bn)

        print("\n" + "=" * 70)
        print("🎉 All new features working correctly!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
