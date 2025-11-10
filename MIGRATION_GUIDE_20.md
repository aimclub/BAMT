# BAMT 2.0.0 Migration Guide

## Overview

BAMT 2.0.0 represents a complete architectural refresh with a clean, sklearn-like API. This guide helps users migrate from v1.x to v2.0.

## Key Changes

### Architecture
- **Old (v1.x)**: `bamt/networks/`, `bamt/builders/`, `bamt/nodes/`
- **New (v2.0)**: `bamt/models/`, `bamt/dag_optimizers/`, `bamt/core/`, `bamt/score_functions/`

### API Style
- **Old**: Method chaining with builders
- **New**: sklearn-like fit/predict/sample interface

## Migration Examples

### Example 1: Continuous Bayesian Network

**Old API (v1.x):**
```python
from bamt.networks.continuous_bn import ContinuousBN

bn = ContinuousBN(use_mixture=True)
bn.add_nodes(descriptor)
bn.add_edges(data, scoring_function=("K2", K2))
bn.fit_parameters(data)
samples = bn.sample(1000, progress_bar=False)
```

**New API (v2.0):**
```python
from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
from bamt.dag_optimizers.score import HillClimbingOptimizer
from bamt.score_functions import K2Score

# Learn structure
optimizer = HillClimbingOptimizer(score_function=K2Score())
edges = optimizer.optimize(data)

# Fit and sample
bn = ContinuousBayesianNetwork()
bn.set_structure(edges)
bn.fit(data)
samples = bn.sample(1000)
```

### Example 2: Discrete Bayesian Network

**Old API (v1.x):**
```python
from bamt.networks.discrete_bn import DiscreteBN

bn = DiscreteBN()
bn.add_nodes(descriptor)
bn.add_edges(data, optimizer="HC")
bn.fit_parameters(data)
```

**New API (v2.0):**
```python
from bamt.models.probabilistic_structural_models import DiscreteBayesianNetwork
from bamt.dag_optimizers.score import HillClimbingOptimizer
from bamt.score_functions import MutualInformationScore

optimizer = HillClimbingOptimizer(score_function=MutualInformationScore())
edges = optimizer.optimize(data)

bn = DiscreteBayesianNetwork()
bn.set_structure(edges)
bn.fit(data)
```

### Example 3: Hybrid Bayesian Network

**Old API (v1.x):**
```python
from bamt.networks.hybrid_bn import HybridBN

bn = HybridBN(has_logit=False, use_mixture=True)
bn.add_nodes(descriptor)
bn.add_edges(data)
bn.fit_parameters(data)
```

**New API (v2.0):**
```python
from bamt.models.probabilistic_structural_models import HybridBayesianNetwork

bn = HybridBayesianNetwork()
bn.set_structure(edges)  # edges from structure learning or expert knowledge
bn.fit(data)  # Automatically infers continuous vs discrete
```

### Example 4: Structure Learning

**Old API (v1.x):**
```python
bn = HybridBN()
bn.add_nodes(descriptor)
bn.add_edges(data, scoring_function=("MI",), optimizer="HC")
```

**New API (v2.0):**
```python
from bamt.dag_optimizers.score import HillClimbingOptimizer
from bamt.score_functions import MutualInformationScore

optimizer = HillClimbingOptimizer(
    score_function=MutualInformationScore(),
    max_iter=100
)
edges = optimizer.optimize(data)
```

## Feature Parity

### ✅ Fully Migrated Features
- Continuous Bayesian Networks
- Discrete Bayesian Networks
- Hybrid Bayesian Networks
- Hill Climbing structure learning
- K2 and MI score functions
- Parameter estimation (MLE-based)
- Sampling from fitted networks
- Basic inference/prediction

### ⚠️ Advanced Features (Optional)
These features exist in v1.x but are not critical for basic usage:
- BigBraveBN (for very large networks)
- CompositeBN (ensemble networks)
- Evolutionary structure learning
- Custom visualization utilities

Users needing these features can continue using v1.x API which remains available.

## Benefits of 2.0.0

1. **Cleaner API**: sklearn-like interface is intuitive and well-documented
2. **Modular Design**: Easy to extend with custom optimizers and score functions
3. **Type Hints**: Better IDE support and type checking
4. **Test Coverage**: Comprehensive test suite with TDD approach
5. **Optional Dependencies**: Graceful degradation when packages unavailable

## Compatibility

Both v1.x and v2.0 APIs coexist in the same package. Code using v1.x will continue to work:

```python
# v1.x code still works
from bamt.networks.continuous_bn import ContinuousBN
bn = ContinuousBN()
# ... existing code ...

# v2.0 code in same project
from bamt.models.probabilistic_structural_models import ContinuousBayesianNetwork
bn2 = ContinuousBayesianNetwork()
# ... new code ...
```

## Recommended Migration Path

1. **Phase 1**: Keep existing v1.x code working
2. **Phase 2**: Write new features using v2.0 API
3. **Phase 3**: Gradually migrate old code module-by-module
4. **Phase 4**: Once fully migrated, consider removing v1.x dependencies

## Getting Help

- Documentation: [Read the Docs](https://bamt.readthedocs.io/)
- Issues: [GitHub Issues](https://github.com/aimclub/BAMT/issues)
- Examples: See `tests/test_20_*.py` for usage examples

## Summary

BAMT 2.0.0 provides a cleaner, more maintainable architecture while preserving all core functionality. The sklearn-like API makes it easier to integrate into existing ML pipelines.
