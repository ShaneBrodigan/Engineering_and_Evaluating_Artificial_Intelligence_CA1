from .Transform import OnehotEncode, LabelEncode, WordEmbeddings, Normalize, Standardize, FrequencyEncode
import pandas as pd

identified_cols ={}

class FeatureEngineering:

    def __init__(self, df):
        self.identified_cols =  self.find_encode_type_for_col(df)
        self.dataframe = df

    ## Chooses Which Column Require Which Encoding Procedure
    def find_encode_type_for_col(self, df):
        cols = df.columns
        identified_cols = {
            "frequency_encode": [],
            "label_encode": [],
            "one_hot_encode": [],
            "word_embedding_encode": [],
            "normalize": [],
            "standardize": [],
        }

        for col in cols:
            # Checks for columns of type 'str'
            if df[col].dtype in ['str', 'object']:
                # Choose col for one hot encode
                if df[col].nunique() < 10:
                    identified_cols["one_hot_encode"].append(col)
                # Choose col for target encode
                elif df[col].nunique() < 20:
                    identified_cols["frequency_encode"].append(col)
                # Choose col for word embeddings
                else:
                    identified_cols["word_embedding_encode"].append(col)

            elif df[col].dtype == 'bool':
                identified_cols["label_encode"].append(col)

            # Checks for columns of type 'int' or 'float'
            if df[col].dtype in ['int', 'float']:
                # Choose col for values with negatives
                if df[col].min() < 0:
                    identified_cols["standardize"].append(col)
                # Choose col for values with just positives
                else:
                    identified_cols["normalize"].append(col)

        return identified_cols


    def process_data(self):
        one_hot = OnehotEncode()
        df = one_hot.transform(self.dataframe)

        frequency_encode = FrequencyEncode()
        df = frequency_encode.transform(df)

        label_encode = LabelEncode()
        df = label_encode.transform(df)

        word_embeddings = WordEmbeddings()
        df = word_embeddings.transform(df)

        normalize = Normalize()
        df = normalize.transform(df)

        standardize = Standardize()
        df = standardize.transform(df)

        return df


    def col_class_splitter(self, col) -> dict:
        df = self.dataframe
        unique_classes = df[col].unique()
        split_dfs = dict.fromkeys(unique_classes)

        for cls in unique_classes:
            split_dfs[cls] = df[df[col] == cls].copy()

        return split_dfs