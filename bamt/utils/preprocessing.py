"""
Preprocessing utilities for BAMT 2.0.0.

Provides data preparation functions for Bayesian Network learning.
"""

from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer


class DataPreprocessor:
    """Preprocessor for Bayesian Network data."""

    @staticmethod
    def infer_column_types(data: pd.DataFrame) -> Dict[str, str]:
        """
        Infer column types from data.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary mapping column names to types ('discrete' or 'continuous')

        Example:
            >>> types = DataPreprocessor.infer_column_types(data)
            >>> print(types)  # {'A': 'discrete', 'B': 'continuous', ...}
        """
        types = {}
        for col in data.columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check if it's integer with few unique values (likely discrete)
                if pd.api.types.is_integer_dtype(data[col]):
                    unique_count = data[col].nunique()
                    if unique_count < 20:  # Heuristic threshold
                        types[col] = "discrete"
                    else:
                        types[col] = "continuous"
                else:
                    # Float type - treat as continuous
                    types[col] = "continuous"
            else:
                # Non-numeric (object/categorical) - treat as discrete
                types[col] = "discrete"

        return types

    @staticmethod
    def encode_discrete_columns(
        data: pd.DataFrame, discrete_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Encode discrete columns using LabelEncoder.

        Args:
            data: Input DataFrame
            discrete_columns: List of discrete column names. If None, auto-inferred.

        Returns:
            Tuple of (encoded_data, encoding_dict)
            encoding_dict maps column names to {original_value: encoded_value}

        Example:
            >>> data_encoded, encoding = DataPreprocessor.encode_discrete_columns(data)
            >>> print(encoding['Category'])  # {'A': 0, 'B': 1, 'C': 2}
        """
        data_encoded = data.copy()

        if discrete_columns is None:
            types = DataPreprocessor.infer_column_types(data)
            discrete_columns = [k for k, v in types.items() if v == "discrete"]

        encoding_dict = {}

        for col in discrete_columns:
            if col not in data.columns:
                continue

            encoder = LabelEncoder()
            data_encoded[col] = encoder.fit_transform(data[col].astype(str))

            # Store encoding mapping
            encoding_dict[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

        return data_encoded, encoding_dict

    @staticmethod
    def discretize_continuous_columns(
        data: pd.DataFrame,
        continuous_columns: Optional[List[str]] = None,
        n_bins: int = 5,
        strategy: str = "quantile",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Discretize continuous columns into bins.

        Args:
            data: Input DataFrame
            continuous_columns: List of continuous column names. If None, auto-inferred.
            n_bins: Number of bins for discretization
            strategy: Discretization strategy ('uniform', 'quantile', 'kmeans')

        Returns:
            Tuple of (discretized_data, discretizer_dict)

        Example:
            >>> data_disc, discretizers = DataPreprocessor.discretize_continuous_columns(
            ...     data, n_bins=3, strategy='quantile'
            ... )
        """
        data_discretized = data.copy()

        if continuous_columns is None:
            types = DataPreprocessor.infer_column_types(data)
            continuous_columns = [k for k, v in types.items() if v == "continuous"]

        discretizer_dict = {}

        for col in continuous_columns:
            if col not in data.columns:
                continue

            discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=strategy
            )

            # Reshape for sklearn
            values = data[col].values.reshape(-1, 1)
            data_discretized[col] = discretizer.fit_transform(values).astype(int).flatten()

            discretizer_dict[col] = {
                "n_bins": n_bins,
                "strategy": strategy,
                "bin_edges": discretizer.bin_edges_[0].tolist(),
            }

        return data_discretized, discretizer_dict

    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame, strategy: str = "drop", fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in data.

        Args:
            data: Input DataFrame
            strategy: Strategy for handling missing values:
                     - 'drop': Drop rows with missing values
                     - 'mean': Fill with column mean (continuous only)
                     - 'median': Fill with column median (continuous only)
                     - 'mode': Fill with column mode (discrete)
                     - 'constant': Fill with specified value
            fill_value: Value to use when strategy='constant'

        Returns:
            DataFrame with missing values handled

        Example:
            >>> data_clean = DataPreprocessor.handle_missing_values(data, strategy='median')
        """
        data_clean = data.copy()

        if strategy == "drop":
            data_clean = data_clean.dropna()
        elif strategy == "mean":
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
            data_clean[numeric_cols] = data_clean[numeric_cols].fillna(
                data_clean[numeric_cols].mean()
            )
        elif strategy == "median":
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
            data_clean[numeric_cols] = data_clean[numeric_cols].fillna(
                data_clean[numeric_cols].median()
            )
        elif strategy == "mode":
            for col in data_clean.columns:
                data_clean[col] = data_clean[col].fillna(data_clean[col].mode()[0])
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value must be specified when strategy='constant'")
            data_clean = data_clean.fillna(fill_value)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return data_clean

    @staticmethod
    def split_columns_by_type(
        data: pd.DataFrame, column_types: Optional[Dict[str, str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Split columns into discrete and continuous lists.

        Args:
            data: Input DataFrame
            column_types: Optional dict mapping column names to types.
                         If None, types are auto-inferred.

        Returns:
            Tuple of (discrete_columns, continuous_columns)

        Example:
            >>> disc_cols, cont_cols = DataPreprocessor.split_columns_by_type(data)
        """
        if column_types is None:
            column_types = DataPreprocessor.infer_column_types(data)

        discrete_columns = [k for k, v in column_types.items() if v == "discrete"]
        continuous_columns = [k for k, v in column_types.items() if v == "continuous"]

        return discrete_columns, continuous_columns
