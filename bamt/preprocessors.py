from typing import Tuple, Dict

from pandas import DataFrame

from bamt.log import logger_preprocessor
from bamt.utils import GraphUtils as gru


class BasePreprocessor(object):
    """
    Base for Preprocessor
    """

    def __init__(self):
        self.nodes_signs = {}
        self.nodes_types = {}

    @staticmethod
    def get_nodes_types(data):
        return gru.nodes_types(data=data)

    def get_nodes_signs(self, data):
        return gru.nodes_signs(nodes_types=self.nodes_types, data=data)

    def code_categories(
        self, data: DataFrame, encoder
    ) -> Tuple[DataFrame, Dict[str, Dict]]:
        """Encoding categorical parameters

        Args:
            data (DataFrame): input dataset
            encoder: any object with fit_transform method

        Returns:
            pd.DataFrame: output dataset with encoded parameters
            dict: dictionary with values and codes
        """
        columns = [
            col for col in data.columns.to_list() if self.nodes_types[col] == "disc"
        ]
        df = data.copy()  # INPUT DF. Debugging SettingWithCopyWarning
        if not columns:
            return df, None
        data = df[columns]  # DATA TO CATEGORIZE
        encoder_dict = dict()

        for col_name, column in data.items():
            # Iterate over (column name, Series) pairs.
            try:
                df[col_name] = encoder.fit_transform(column.values)
            except TypeError as exc:
                logger_preprocessor.error(
                    f"Wrond data types on {col_name} ({df[col_name].dtypes}). Message: {exc}"
                )
            try:
                mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
                encoder_dict[col_name] = mapping
            except BaseException:
                pass
        return df, encoder_dict

    def discretize(self, data: DataFrame, discretizer) -> tuple:
        columns = [
            col for col in data.columns.to_list() if self.nodes_types[col] == "cont"
        ]
        df = data.copy()
        if not columns:
            return df, None
        data = df[columns]

        data_discrete = discretizer.fit_transform(data.values)
        df[columns] = data_discrete.astype("int")

        return df, discretizer

    def decode(self):
        pass


class Preprocessor(BasePreprocessor):
    def __init__(self, pipeline: list):
        super().__init__()
        assert isinstance(pipeline, list), "pipeline must be list"
        self.pipeline = pipeline
        self.coder = None

    @property
    def info(self):
        return {"types": self.nodes_types, "signs": self.nodes_signs}

    def scan(self, data: DataFrame):
        """
        Function to scan data. If something is wrong, it will be send to log file
        """
        columns_cont = [
            col for col in data.columns.to_list() if self.nodes_types[col] == "cont"
        ]
        if not columns_cont:
            logger_preprocessor.info("No one column is continuous")

        columns_disc = [
            col
            for col in data.columns.to_list()
            if self.nodes_types[col] in ["disc", "disc_num"]
        ]
        if not columns_disc:
            logger_preprocessor.info("No one column is discrete")

    def apply(self, data: DataFrame) -> Tuple[DataFrame, Dict]:
        """
        Apply pipeline
        data: data to apply on
        """
        df = data.copy()
        self.nodes_types = self.get_nodes_types(data)
        if list(self.nodes_types.keys()) != data.columns.to_list():
            logger_preprocessor.error("Nodes_types dictionary are not full.")
            return None, None
        self.nodes_signs = self.get_nodes_signs(data)
        self.scan(df)
        for name, instrument in self.pipeline:
            if name == "encoder":
                df, self.coder = self.code_categories(data=data, encoder=instrument)
            if name == "discretizer":
                df, est = self.discretize(data=df, discretizer=instrument)
        return df, self.coder
