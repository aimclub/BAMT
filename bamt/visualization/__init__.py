"""
Visualization utilities for BAMT 2.0.0
Provides simple plotting functions for Bayesian networks
"""

try:
    from .plot import plot_structure, plot_network_info
    __all__ = ['plot_structure', 'plot_network_info']
except ImportError:
    # If dependencies not available, export empty list
    __all__ = []
