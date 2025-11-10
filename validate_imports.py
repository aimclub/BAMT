#!/usr/bin/env python3
"""
Validation script to check if all new BAMT 2.0.0 modules can be imported correctly.
"""

def test_core_imports():
    """Test core module imports."""
    print("Testing core module imports...")
    
    # Test graph imports
    try:
        from bamt.core.graph import Graph, DAG
        print("✓ core.graph imports successful")
    except ImportError as e:
        print(f"✗ core.graph import failed: {e}")
        return False
    
    # Test node_models imports
    try:
        from bamt.core.node_models import (
            Classifier, 
            Regressor,
            ContinuousDistribution,
            EmpiricalDistribution
        )
        print("✓ core.node_models imports successful")
    except ImportError as e:
        print(f"✗ core.node_models import failed: {e}")
        return False
    
    # Test nodes imports
    try:
        from bamt.core.nodes import (
            DiscreteNode,
            ContinuousNode,
            ConditionalDiscreteNode,
            ConditionalContinuousNode
        )
        print("✓ core.nodes imports successful")
    except ImportError as e:
        print(f"✗ core.nodes import failed: {e}")
        return False
    
    return True


def test_dag_optimizers_imports():
    """Test dag_optimizers module imports."""
    print("\nTesting dag_optimizers module imports...")
    
    try:
        from bamt.dag_optimizers import DAGOptimizer
        print("✓ dag_optimizers base import successful")
    except ImportError as e:
        print(f"✗ dag_optimizers import failed: {e}")
        return False
    
    return True


def test_score_functions_imports():
    """Test score_functions module imports."""
    print("\nTesting score_functions module imports...")
    
    try:
        from bamt.score_functions import ScoreFunction, K2Score, MutualInformationScore
        print("✓ score_functions imports successful")
    except ImportError as e:
        print(f"✗ score_functions import failed: {e}")
        return False
    
    return True


def test_parameter_estimators_imports():
    """Test parameter_estimators module imports."""
    print("\nTesting parameter_estimators module imports...")
    
    try:
        from bamt.parameter_estimators import ParametersEstimator, MaximumLikelihoodEstimator
        print("✓ parameter_estimators imports successful")
    except ImportError as e:
        print(f"✗ parameter_estimators import failed: {e}")
        return False
    
    return True


def test_models_imports():
    """Test models module imports."""
    print("\nTesting models module imports...")
    
    try:
        from bamt.models.probabilistic_structural_models import (
            ProbabilisticStructuralModel,
            BayesianNetwork,
            ContinuousBayesianNetwork,
            DiscreteBayesianNetwork,
            HybridBayesianNetwork,
            CompositeBayesianNetwork
        )
        print("✓ models.probabilistic_structural_models imports successful")
    except ImportError as e:
        print(f"✗ models import failed: {e}")
        return False
    
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("BAMT 2.0.0 Import Validation")
    print("=" * 60)
    
    results = []
    results.append(("Core modules", test_core_imports()))
    results.append(("DAG optimizers", test_dag_optimizers_imports()))
    results.append(("Score functions", test_score_functions_imports()))
    results.append(("Parameter estimators", test_parameter_estimators_imports()))
    results.append(("Models", test_models_imports()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All imports successful!")
        return 0
    else:
        print("\n✗ Some imports failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
