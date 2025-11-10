# Phase 5 Completion - Advanced Features & Utilities

## Overview

Phase 5 successfully adds advanced features and utilities to complete the BAMT 2.0.0 architecture migration.

## Deliverables

### 1. Score Functions ✅

**BICScore (Bayesian Information Criterion)**
- Log-likelihood with strong complexity penalty
- Formula: `BIC = LL - (k/2) * log(n)`
- Best for larger datasets
- File: `bamt/score_functions/bic_score.py`

**AICScore (Akaike Information Criterion)**
- Log-likelihood with moderate complexity penalty
- Formula: `AIC = LL - k`
- Best for smaller datasets
- File: `bamt/score_functions/aic_score.py`

**Usage:**
```python
from bamt.score_functions import BICScore, AICScore
from bamt.dag_optimizers.score import HillClimbingOptimizer

# Use BIC for structure learning
optimizer = HillClimbingOptimizer(score_function=BICScore())
edges = optimizer.optimize(data)
```

### 2. Visualization Utilities ✅

**plot_structure()**
- Visualize DAG structure using networkx
- Customizable node colors, sizes, labels
- Save to file support

**plot_network_info()**
- Display network structure + statistics
- Shows in-degree, out-degree, root/leaf nodes
- Two-panel layout

**Usage:**
```python
from bamt.visualization import plot_structure, plot_network_info

# Plot just structure
fig = plot_structure(edges, title="My Network")

# Plot with statistics
fig = plot_network_info(bn, title="Network Analysis")
plt.show()
```

**Files:**
- `bamt/visualization/plot.py`
- `bamt/visualization/__init__.py`

### 3. Network Utilities ✅

**save_network() / load_network()**
- Save/load in JSON or pickle format
- JSON stores structure only
- Pickle stores full fitted model

**network_to_dict()**
- Convert network to dictionary
- Useful for serialization

**Usage:**
```python
from bamt.utils_20 import save_network, load_network, network_to_dict

# Save structure as JSON
save_network(bn, 'network.json', format='json')

# Save fitted model as pickle
save_network(bn, 'network.pkl', format='pickle')

# Load
loaded_bn = load_network('network.json', format='json')

# Export to dict
data = network_to_dict(bn)
```

**Files:**
- `bamt/utils_20/network_utils.py`
- `bamt/utils_20/__init__.py`

## Test Coverage

### New Tests in Phase 5:

1. **test_20_bic_aic_scores.py** (6 tests)
   - BIC initialization and computation
   - AIC initialization and computation
   - Complexity penalty validation

2. **test_20_visualization.py** (2 tests)
   - Structure plotting
   - Network info plotting

3. **test_20_utils.py** (3 tests)
   - Save/load JSON format
   - Save/load different network types
   - Network to dict conversion

4. **test_20_phase5_integration.py** (4 tests)
   - Complete workflow with BIC
   - Complete workflow with AIC
   - Score function comparison
   - Pickle save with fitted parameters

**Total Phase 5 Tests:** 15 tests

## Complete Feature Matrix

| Category | Features | Phase | Status |
|----------|----------|-------|--------|
| **Networks** | ContinuousBN, DiscreteBN, HybridBN | 1-3 | ✅ |
| **Optimizers** | Hill Climbing | 2 | ✅ |
| **Score Functions** | K2, MI | 2 | ✅ |
| **Score Functions** | **BIC, AIC** | **5** | **✅** |
| **Visualization** | **plot_structure, plot_info** | **5** | **✅** |
| **Utilities** | **save, load, export** | **5** | **✅** |
| **Core Operations** | fit, predict, sample | 1-3 | ✅ |

## Integration Points

### With Structure Learning
```python
# All 4 score functions work with Hill Climbing
from bamt.score_functions import K2Score, MutualInformationScore, BICScore, AICScore

optimizers = [
    HillClimbingOptimizer(score_function=K2Score()),
    HillClimbingOptimizer(score_function=MutualInformationScore()),
    HillClimbingOptimizer(score_function=BICScore()),
    HillClimbingOptimizer(score_function=AICScore()),
]
```

### With Network Training
```python
# Save/load works with all network types
from bamt.models.probabilistic_structural_models import (
    ContinuousBayesianNetwork,
    DiscreteBayesianNetwork,
    HybridBayesianNetwork
)

networks = [ContinuousBayesianNetwork(), DiscreteBayesianNetwork(), HybridBayesianNetwork()]
for bn in networks:
    bn.set_structure(edges)
    bn.fit(data)
    save_network(bn, f'{bn.__class__.__name__}.pkl', format='pickle')
```

### With Visualization
```python
# Visualize any network type
plot_network_info(continuous_bn)
plot_network_info(discrete_bn)
plot_network_info(hybrid_bn)
```

## Code Quality

- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Error handling
- ✅ Optional dependencies
- ✅ Test-driven development
- ✅ Code review validated

## Performance Characteristics

### Score Functions
- **K2**: O(n * k) where k = number of parameters
- **MI**: O(n * e) where e = number of edges
- **BIC**: O(n * k) with log(n) penalty
- **AIC**: O(n * k) with linear penalty

### Visualization
- **plot_structure**: O(n + e) using networkx
- **plot_network_info**: O(n + e) + statistics computation

### Utilities
- **JSON save/load**: O(n + e) - fast, portable
- **Pickle save/load**: O(model_size) - slower but preserves everything

## Backward Compatibility

All Phase 5 features are additive:
- ✅ No breaking changes
- ✅ v1.x API still works
- ✅ Optional dependencies
- ✅ Graceful degradation

## Summary

Phase 5 successfully completes the BAMT 2.0.0 migration with:
- ✅ 2 new score functions (BIC, AIC)
- ✅ 2 visualization functions
- ✅ 3 utility functions
- ✅ 15 comprehensive tests
- ✅ Full documentation

**Total Implementation:**
- 5 Phases completed
- 17 implementation files
- 7 test files with 29+ tests
- 3 documentation files
- ~2,500 lines of new code

**Status: Phase 5 Complete ✅**

All planned features for BAMT 2.0.0 architecture migration have been successfully implemented and tested.
