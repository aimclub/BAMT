from Utils import GraphUtils as gru
from log import logger_preprocessor


class BasePreprocessor(object):
    def __init__(self):
        self.nodes_signs = {}
        self.nodes_types = {}

    @staticmethod
    def get_nodes_types(data):
        return gru.nodes_types(data=data)

    def get_nodes_signs(self, data):
        return gru.nodes_signs(nodes_types=self.nodes_types,
                               data=data)

    def code_categories(self, data, encoder):
        """Encoding categorical parameters

            Args:
                data (pd.DataFrame): input dataset
                encoder: any object with fit_transform method

            Returns:
                pd.DataFrame: output dataset with encoded parameters
                dict: dictionary with values and codes
            """
        columns = [col for col in data.columns.to_list() if self.nodes_types[col] == 'disc']
        df = data.copy()  # INPUT DF. Debugging SettingWithCopyWarning
        if not columns:
            logger_preprocessor.info("No one column is discrete")
            return df, None
        data = df[columns]  # DATA TO CATEGORIZE
        encoder_dict = dict()

        for col_name, column in data.iteritems():
            # Iterate over (column name, Series) pairs.
            try:
                df[col_name] = encoder.fit_transform(column.values)
            except TypeError as exc:
                logger_preprocessor.error(f"Wrond data types on {col_name} ({df[col_name].dtypes}). Message: {exc}")
            try:
                mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
                encoder_dict[col_name] = mapping
            except:
                pass
        return df, encoder_dict

    def discretize(self, data, discretizer):
        columns = [col for col in data.columns.to_list() if self.nodes_types[col] == 'cont']
        df = data.copy()
        if not columns:
            logger_preprocessor.info("No one column is continuous")
            return df, None
        data = df[columns]

        data_discrete = discretizer.fit_transform(data.values)
        df[columns] = data_discrete.astype('int')

        return df, discretizer

    def decode(self):
        pass


class Preprocessor(BasePreprocessor):
    def __init__(self, pipeline):
        super().__init__()
        assert isinstance(pipeline, list)
        self.pipeline = pipeline
        self.coder = {}

    @property
    def info(self):
        return {'types': self.nodes_types, 'signs': self.nodes_signs}

    def apply(self, data, dropna=True):
        if dropna:
            data = data.dropna()
            data.reset_index(inplace=True, drop=True)
        df = data.copy()
        self.nodes_types = self.get_nodes_types(data)
        self.nodes_signs = self.get_nodes_signs(data)
        for name, instrument in self.pipeline:
            if name == 'encoder':
                df, self.coder = self.code_categories(data=data, encoder=instrument)
            if name == 'discretizer':
                df, est = self.discretize(data=df, discretizer=instrument)
        return df, self.coder
