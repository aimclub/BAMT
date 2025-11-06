#!/usr/bin/env python3
"""
Module structure validation script for BAMT 2.0.0.
This checks that all expected files and classes exist without requiring dependencies.
"""

import os
import ast
from pathlib import Path


def extract_class_names(file_path):
    """Extract class names from a Python file using AST."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    except Exception as e:
        return []


def check_module_structure():
    """Check that all expected modules and classes exist."""
    print("=" * 70)
    print("BAMT 2.0.0 Module Structure Validation")
    print("=" * 70)
    
    # Expected module structure
    expected_structure = {
        'bamt/core/graph': {
            'graph.py': ['Graph'],
            'dag.py': ['DirectedAcyclicGraph'],
            '__init__.py': []
        },
        'bamt/core/node_models': {
            'distribution.py': ['Distribution'],
            'empirical_distribution.py': ['EmpiricalDistribution'],
            'continuous_distribution.py': ['ContinuousDistribution', 'DistributionPool'],
            'prediction_model.py': ['PredictionModel'],
            'classifier.py': ['Classifier'],
            'regressor.py': ['Regressor'],
            '__init__.py': []
        },
        'bamt/core/nodes': {
            'node.py': ['Node'],
            '__init__.py': []
        },
        'bamt/core/nodes/root_nodes': {
            'root_node.py': ['RootNode'],
            'discrete_node.py': ['DiscreteNode'],
            'continuous_node.py': ['ContinuousNode'],
            '__init__.py': []
        },
        'bamt/core/nodes/child_nodes': {
            'child_node.py': ['ChildNode'],
            'conditional_discrete_node.py': ['ConditionalDiscreteNode'],
            'conditional_continuous_node.py': ['ConditionalContinuousNode'],
            '__init__.py': []
        },
        'bamt/dag_optimizers': {
            'dag_optimizer.py': ['DAGOptimizer'],
            '__init__.py': []
        },
        'bamt/score_functions': {
            'score_function.py': ['ScoreFunction'],
            'k2_score.py': ['K2Score'],
            'mutual_information_score.py': ['MutualInformationScore'],
            '__init__.py': []
        },
        'bamt/parameter_estimators': {
            'parameters_estimator.py': ['ParametersEstimator'],
            'maximum_likelihood_estimator.py': ['MaximumLikelihoodEstimator'],
            '__init__.py': []
        },
        'bamt/models/probabilistic_structural_models': {
            'probabilistic_structural_model.py': ['ProbabilisticStructuralModel'],
            'bayesian_network.py': ['BayesianNetwork'],
            'continuous_bayesian_network.py': ['ContinuousBayesianNetwork'],
            'discrete_bayesian_network.py': ['DiscreteBayesianNetwork'],
            'hybrid_bayesian_network.py': ['HybridBayesianNetwork'],
            'composite_bayesian_network.py': ['CompositeBayesianNetwork'],
            '__init__.py': []
        }
    }
    
    all_valid = True
    
    for module_path, files in expected_structure.items():
        print(f"\n📦 {module_path}")
        
        if not os.path.exists(module_path):
            print(f"  ✗ Module directory not found!")
            all_valid = False
            continue
        
        for file_name, expected_classes in files.items():
            file_path = os.path.join(module_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"  ✗ {file_name} - File not found")
                all_valid = False
                continue
            
            if expected_classes:
                found_classes = extract_class_names(file_path)
                
                # Check if all expected classes are present
                missing_classes = set(expected_classes) - set(found_classes)
                
                if missing_classes:
                    print(f"  ✗ {file_name} - Missing classes: {missing_classes}")
                    all_valid = False
                else:
                    print(f"  ✓ {file_name} - Contains {', '.join(expected_classes)}")
            else:
                print(f"  ✓ {file_name} - Exists")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    if all_valid:
        print("\n✓ All expected modules and classes exist!")
        print("\nNote: This validation checks file and class existence only.")
        print("Full import validation requires all dependencies to be installed.")
        return 0
    else:
        print("\n✗ Some modules or classes are missing.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(check_module_structure())
