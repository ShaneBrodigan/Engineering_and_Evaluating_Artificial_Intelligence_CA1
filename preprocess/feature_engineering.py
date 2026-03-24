from .Transform import OnehotEncode, LabelEncode, WordEmbeddings, Normalize, Standardize, FrequencyEncode
from config import Config
import pandas as pd

identified_cols ={}

class FeatureEngineering:

    def __init__(self, df):
        self.config = Config()
        self.identified_cols = self.find_encode_type_for_col(df)
        self.dataframe = df

    ## Chooses Which Column Require Which Encoding Procedure
    def find_encode_type_for_col(self, df):
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
            # Checks for columns of type 'str'
            if df[col].dtype in ['str', 'object']:
                # Choose col for one hot encode
                if df[col].nunique() < 4:
                    identified_cols["ONE_HOT_COLS"].append(col)
                # Choose col for target encode
                elif df[col].nunique() < 25:
                    identified_cols["FREQUENCY_ENCODE_COLS"].append(col)
                # Choose col for word embeddings
                else:
                    identified_cols["WORD_EMBEDDING_COLS"].append(col)

            elif df[col].dtype == 'bool':
                identified_cols["LABEL_ENCODE_COLS"].append(col)

            # Checks for columns of type 'int' or 'float'
            if df[col].dtype in ['int', 'float']:
                # Choose col for values with negatives
                if df[col].min() < 0:
                    identified_cols["STANDARDIZE_ENCODE_COLS"].append(col)
                # Choose col for values with just positives
                else:
                    identified_cols["NORMALIZE_ENCODE_COLS"].append(col)

        print(identified_cols)
        return identified_cols


    def process_data(self):
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
        df = self.dataframe
        unique_classes = df[col].unique()
        split_dfs = dict.fromkeys(unique_classes)

        for cls in unique_classes:
            split_dfs[cls] = df[df[col] == cls].copy()

        return split_dfs