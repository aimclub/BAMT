from bamt.utils.math_utils import get_brave_matrix, get_proximity_matrix


class BigBraveBN:
    def __init__(self, n_nearest=5, threshold=0.3, proximity_metric="MI"):
        self.n_nearest = n_nearest
        self.threshold = threshold
        self.proximity_metric = proximity_metric
        self.possible_edges = []

    def set_possible_edges_by_brave(self, df):
        """Returns list of possible edges for structure learning

        Args:
            df (DataFrame): data

        Returns:
            Possible edges: list of possible edges
        """

        proximity_matrix = get_proximity_matrix(
            df, proximity_metric=self.proximity_metric
        )
        brave_matrix = get_brave_matrix(df.columns, proximity_matrix, self.n_nearest)

        possible_edges_list = []

        for c1 in df.columns:
            for c2 in df.columns:
                if (
                    brave_matrix.loc[c1, c2]
                    > brave_matrix.max(numeric_only=True).max() * self.threshold
                ):
                    possible_edges_list.append((c1, c2))

        self.possible_edges = possible_edges_list
