import math

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OrdinalEncoder


class BigBraveBN:
    def __init__(self):
        self.possible_edges = []

    def set_possible_edges_by_brave(
        self,
        df: pd.DataFrame,
        n_nearest: int = 5,
        threshold: float = 0.3,
        proximity_metric: str = "MI",
    ) -> list:
        """Returns list of possible edges for structure learning and sets it into attribute

        Args:
            df (pd.DataFrame): Data.
            n_nearest (int): Number of nearest neighbors to consider. Default is 5.
            threshold (float): Threshold for selecting edges. Default is 0.3.
            proximity_metric (str): Metric used to calculate proximity. Default is "MI".

        Returns:
            None: Modifies the object's possible_edges attribute.
        """
        df_copy = df.copy(deep=True)
        proximity_matrix = self._get_proximity_matrix(df_copy, proximity_metric)
        brave_matrix = self._get_brave_matrix(df_copy.columns, proximity_matrix, n_nearest)

        threshold_value = brave_matrix.max(numeric_only=True).max() * threshold
        filtered_brave_matrix = brave_matrix[brave_matrix > threshold_value].stack()
        self.possible_edges = filtered_brave_matrix.index.tolist()
        return self.possible_edges

    @staticmethod
    def _get_n_nearest(
        data: pd.DataFrame, columns: list, corr: bool = False, number_close: int = 5
    ) -> list:
        """Returns N nearest neighbors for every column of dataframe."""
        groups = []
        for c in columns:
            close_ind = data[c].sort_values(ascending=not corr).index.tolist()
            groups.append(close_ind[: number_close + 1])
        return groups

    @staticmethod
    def _get_proximity_matrix(df: pd.DataFrame, proximity_metric: str) -> pd.DataFrame:
        """Returns matrix of proximity for the dataframe."""
        encoder = OrdinalEncoder()
        df_coded = df.copy()
        columns_to_encode = list(df_coded.select_dtypes(include=["category", "object"]))
        df_coded[columns_to_encode] = encoder.fit_transform(df_coded[columns_to_encode])

        if proximity_metric == "MI":
            df_distance = pd.DataFrame(
                np.zeros((len(df.columns), len(df.columns))),
                columns=df.columns,
                index=df.columns,
            )
            for c1 in df.columns:
                for c2 in df.columns:
                    dist = mutual_info_score(df_coded[c1].values, df_coded[c2].values)
                    df_distance.loc[c1, c2] = dist
            return df_distance

        elif proximity_metric == "pearson":
            return df_coded.corr(method="pearson")

    def _get_brave_matrix(
        self, df_columns: pd.Index, proximity_matrix: pd.DataFrame, n_nearest: int = 5
    ) -> pd.DataFrame:
        """Returns matrix of Brave coefficients for the DataFrame."""
        brave_matrix = pd.DataFrame(
            np.zeros((len(df_columns), len(df_columns))),
            columns=df_columns,
            index=df_columns,
        )
        groups = self._get_n_nearest(
            proximity_matrix, df_columns.tolist(), corr=True, number_close=n_nearest
        )

        for c1 in df_columns:
            for c2 in df_columns:
                a = b = c = d = 0.0
                if c1 != c2:
                    for g in groups:
                        a += (c1 in g) & (c2 in g)
                        b += (c1 in g) & (c2 not in g)
                        c += (c1 not in g) & (c2 in g)
                        d += (c1 not in g) & (c2 not in g)

                    divisor = (math.sqrt((a + c) * (b + d))) * (
                        math.sqrt((a + b) * (c + d))
                    )
                    br = (a * len(groups) + (a + c) * (a + b)) / (
                        divisor if divisor != 0 else 0.0000000001
                    )
                    brave_matrix.loc[c1, c2] = br

        return brave_matrix
