from .Transform import OnehotEncode, LabelEncode, WordEmbeddings, Normalize, Standardize, FrequencyEncode
from config import Config

identified_cols ={}

class FeatureEngineering:

    def __init__(self, df):
        self.config = Config()
        self.identified_cols = self.find_encode_type_for_col(df)
        self.dataframe = df

    def find_encode_type_for_col(self, df):
        """Identified the appropriate encoding strategy for each column"""
        model_target_cols = self.config.model_target_cols
        cols = df.drop(columns=model_target_cols, errors='ignore').columns
        identified_cols = {
            "FREQUENCY_ENCODE_COLS": [],
            "LABEL_ENCODE_COLS": [col for col in model_target_cols if col in df.columns],
            "ONE_HOT_COLS": [],
            "WORD_EMBEDDING_COLS": [],
            "NORMALIZE_ENCODE_COLS": [],
            "STANDARDIZE_ENCODE_COLS": [],
        }

        for col in cols:
            if df[col].dtype in ['str', 'object']:
                if df[col].nunique() < 4:
                    identified_cols["ONE_HOT_COLS"].append(col)
                elif df[col].nunique() < 25:
                    identified_cols["FREQUENCY_ENCODE_COLS"].append(col)
                else:
                    identified_cols["WORD_EMBEDDING_COLS"].append(col)

            elif df[col].dtype == 'bool':
                identified_cols["LABEL_ENCODE_COLS"].append(col)

            if df[col].dtype in ['int', 'float']:
                if df[col].min() < 0:
                    identified_cols["STANDARDIZE_ENCODE_COLS"].append(col)
                else:
                    identified_cols["NORMALIZE_ENCODE_COLS"].append(col)

        print(identified_cols)
        return identified_cols


    def process_data(self):
        """Applied all encoding transformations and returned the encoded dataframe alongside the label encoder"""
        one_hot = OnehotEncode(self.identified_cols["ONE_HOT_COLS"])
        df = one_hot.transform(self.dataframe)

        frequency_encode = FrequencyEncode(self.identified_cols["FREQUENCY_ENCODE_COLS"])
        df = frequency_encode.transform(df)

        label_encode = LabelEncode(self.identified_cols["LABEL_ENCODE_COLS"])
        df = label_encode.transform(df)

        word_embeddings = WordEmbeddings(self.identified_cols["WORD_EMBEDDING_COLS"])
        df = word_embeddings.transform(df)

        normalize = Normalize(self.identified_cols["NORMALIZE_ENCODE_COLS"])
        df = normalize.transform(df)

        standardize = Standardize(self.identified_cols["STANDARDIZE_ENCODE_COLS"])
        df = standardize.transform(df)

        return df, label_encode


    def col_class_splitter(self, col) -> dict:
        """Split the dataframe into a dict of sub-dataframes keyed by unique class values in the specified column"""
        df = self.dataframe
        unique_classes = df[col].unique()
        split_dfs = dict.fromkeys(unique_classes)

        for cls in unique_classes:
            split_dfs[cls] = df[df[col] == cls].copy()

        return split_dfs