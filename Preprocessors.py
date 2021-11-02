import logging.config

from os import path

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')

logging.config.fileConfig(log_file_path)
logger = logging.getLogger('preprocessor')


class BasePreprocessor(object):
    def __init__(self):
        self.nodes_signs = {}
        self.nodes_types = {}

    @staticmethod
    def get_nodes_types(data):
        """Function to define the type of the node
               disc - discrete node
               cont - continuous
            Args:
                data (pd.DataFrame): input dataset

            Returns:
                dict: output dictionary where 'key' - node name and 'value' - node type
            """
        column_type = dict()
        for c in data.columns.to_list():
            disc = ['str', 'O', 'b']
            disc_numerical = ['int32', 'int64']
            cont = ['float32', 'float64']
            if data[c].dtypes in disc:
                column_type[c] = 'disc'
            elif data[c].dtype in cont:
                column_type[c] = 'cont'
            elif data[c].dtype in disc_numerical:
                column_type[c] = 'disc_num'
            else:
                logger.error(f'Unsupported data type. Dtype: {data[c].dtypes}')

        return column_type

    def get_nodes_signs(self, data):
        """Function to define sign of the node
               neg - if node has negative values
               pos - if node has only positive values

            Args:
                data (pd.DataFrame): input dataset

            Returns:
                dict: output dictionary where 'key' - node name and 'value' - sign of data
            """
        columns_sign = dict()
        for c in data.columns.to_list():
            if self.nodes_types[c] == 'cont':
                if (data[c] < 0).any():
                    columns_sign[c] = 'neg'
                else:
                    columns_sign[c] = 'pos'
        return columns_sign

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
            logger.warning("No one column is discrete")
            return df, None
        data = df[columns]  # DATA TO CATEGORIZE
        encoder_dict = dict()

        for col_name, column in data.iteritems():
            # Iterate over (column name, Series) pairs.
            try:
                df[col_name] = encoder.fit_transform(column.values)
            except TypeError as exc:
                logger.error(f"Wrond data types on {col_name} ({df[col_name].dtypes}). Message: {exc}")
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
            logger.warning("No one column is continuous")
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
        self.pipeline = pipeline
        self.coder = {}

    @property
    def info(self):
        return {'types': self.nodes_types, 'signs': self.nodes_signs}

    def apply(self, data, dropna=True):
        if dropna:
            data = data.dropna()
            data.reset_index(inplace=True, drop=True)
        self.nodes_types = self.get_nodes_types(data)
        self.nodes_signs = self.get_nodes_signs(data)

        coded_data, self.coder = self.code_categories(data=data, encoder=self.pipeline[0][1])
        discrete_data, est = self.discretize(data=coded_data, discretizer=self.pipeline[1][1])
        return discrete_data, self.coder
