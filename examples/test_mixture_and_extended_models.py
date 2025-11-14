"""
Example demonstrating new features in BAMT 2.0.0:
- Mixture Gaussian distributions for multimodal continuous variables
- Extended ML model repository with XGBoost, CatBoost, LightGBM
- Custom model selection for classifiers and regressors
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from bamt.core.node_models import (
    MixtureGaussianDistribution,
    Classifier,
    Regressor,
)
from bamt.utils.ml_models import get_ml_repository

print("=" * 70)
print("BAMT 2.0.0 - New Features Demonstration")
print("=" * 70)

# ==============================================================================
# 1. Mixture Gaussian Distribution
# ==============================================================================
print("\n1. MIXTURE GAUSSIAN DISTRIBUTION")
print("-" * 70)

# Create bimodal data (two Gaussian peaks)
np.random.seed(42)
data_mode1 = np.random.normal(-2, 0.5, 500)
data_mode2 = np.random.normal(2, 0.5, 500)
bimodal_data = np.concatenate([data_mode1, data_mode2])
np.random.shuffle(bimodal_data)

print(f"Generated bimodal data with {len(bimodal_data)} samples")

# Fit mixture distribution
mixture_dist = MixtureGaussianDistribution(
    component_selection_method="aic_bic_average"
)
mixture_dist.fit(bimodal_data.reshape(-1, 1))

params = mixture_dist.get_parameters()
print(f"Detected {params['n_components']} components")
print(f"Component means: {[round(m[0], 2) for m in params['mean']]}")
print(f"Component weights: {[round(w, 2) for w in params['coef']]}")

# Sample from the distribution
samples = mixture_dist.sample(10)
print(f"Sampled {len(samples)} values: {[round(s[0], 2) for s in samples[:5]]}...")

# ==============================================================================
# 2. Extended ML Models Repository
# ==============================================================================
print("\n2. EXTENDED ML MODELS REPOSITORY")
print("-" * 70)

repo = get_ml_repository()

print("Available regressors:")
for name in repo.list_available_models("regressor"):
    print(f"  - {name}")

print("\nAvailable classifiers:")
for name in repo.list_available_models("classifier"):
    print(f"  - {name}")

# Check for advanced models
print("\nAdvanced models availability:")
print(f"  - XGBoost: {repo.is_model_available('XGBRegressor')}")
print(f"  - CatBoost: {repo.is_model_available('CatBoostRegressor')}")
print(f"  - LightGBM: {repo.is_model_available('LGBMRegressor')}")

# Get a model with default parameters
if repo.is_model_available('XGBRegressor'):
    xgb_params = repo.get_default_params('XGBRegressor')
    print(f"\nXGBoost default params: {xgb_params}")
    xgb_model = repo.get_model('XGBRegressor', **xgb_params)
    print(f"Created model: {xgb_model}")

# ==============================================================================
# 3. Custom Model Selection with Classifier
# ==============================================================================
print("\n3. CUSTOM MODEL SELECTION FOR CLASSIFICATION")
print("-" * 70)

# Generate classification data
from sklearn.datasets import make_classification
X_class, y_class = make_classification(
    n_samples=200, n_features=4, n_informative=3,
    n_redundant=1, n_classes=2, random_state=42
)

# Get custom candidate models
custom_classifiers = {}
custom_classifiers["LogisticRegression"] = repo.get_model(
    "LogisticRegression", max_iter=1000, random_state=42
)
custom_classifiers["RandomForest"] = repo.get_model(
    "RandomForestClassifier", n_estimators=50, random_state=42
)

# Try to add XGBoost if available
if repo.is_model_available('XGBClassifier'):
    custom_classifiers["XGBoost"] = repo.get_model(
        "XGBClassifier", n_estimators=50, random_state=42
    )

# Create classifier with auto-selection
classifier = Classifier(candidate_models=custom_classifiers, cv_folds=3)
classifier.fit(X_class, y_class)

model_info = classifier.get_best_model_info()
print(f"Best model selected: {model_info['best_model']}")
print(f"CV scores: {model_info['cv_scores']}")

# Make predictions
predictions = classifier.predict(X_class[:5])
probabilities = classifier.predict_proba(X_class[:5])
print(f"Sample predictions: {predictions}")
print(f"Sample probabilities (first 2): {probabilities[:2]}")

# ==============================================================================
# 4. Custom Model Selection with Regressor
# ==============================================================================
print("\n4. CUSTOM MODEL SELECTION FOR REGRESSION")
print("-" * 70)

# Generate regression data
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(
    n_samples=200, n_features=4, noise=10.0, random_state=42
)

# Get custom candidate models
custom_regressors = {}
custom_regressors["LinearRegression"] = repo.get_model("LinearRegression")
custom_regressors["RandomForest"] = repo.get_model(
    "RandomForestRegressor", n_estimators=50, random_state=42
)

# Try to add XGBoost if available
if repo.is_model_available('XGBRegressor'):
    custom_regressors["XGBoost"] = repo.get_model(
        "XGBRegressor", n_estimators=50, random_state=42
    )

# Create regressor with auto-selection
regressor = Regressor(candidate_models=custom_regressors, cv_folds=3)
regressor.fit(X_reg, y_reg)

model_info = regressor.get_best_model_info()
print(f"Best model selected: {model_info['best_model']}")
print(f"CV scores (negative MSE): {model_info['cv_scores']}")

# Make predictions
predictions = regressor.predict(X_reg[:5])
print(f"Sample predictions: {[round(p, 2) for p in predictions]}")
print(f"Actual values:      {[round(a, 2) for a in y_reg[:5]]}")

# ==============================================================================
# 5. Conditional Mixture Distribution
# ==============================================================================
print("\n5. CONDITIONAL MIXTURE DISTRIBUTION")
print("-" * 70)

# Create data with parent variable
np.random.seed(42)
n_samples = 200
parent_var = np.random.uniform(-1, 1, n_samples)

# Child variable depends on parent (conditional bimodal)
child_var = np.where(
    parent_var > 0,
    np.random.normal(parent_var * 2, 0.3, n_samples),
    np.random.normal(parent_var * 2, 0.3, n_samples)
)

# Fit conditional mixture
cond_mixture = MixtureGaussianDistribution(
    component_selection_method="bic"
)
cond_mixture.fit(child_var.reshape(-1, 1), parent_values=parent_var.reshape(-1, 1))

print(f"Fitted conditional mixture with {cond_mixture.get_parameters()['n_components']} components")

# Sample conditionally
test_parents = np.array([[0.5], [-0.5], [0.0]])
cond_samples = cond_mixture.sample(n_samples=3, parent_values=test_parents)
cond_preds = cond_mixture.predict(parent_values=test_parents)

print(f"Conditional samples for parents [0.5, -0.5, 0.0]:")
print(f"  Samples: {[round(s[0], 2) for s in cond_samples]}")
print(f"  Predictions: {[round(p[0], 2) for p in cond_preds]}")

# ==============================================================================
print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)
