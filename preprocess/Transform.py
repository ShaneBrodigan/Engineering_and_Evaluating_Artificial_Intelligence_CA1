from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

WORD_EMBEDDING_COLS = ["interaction_content", "ticket_summary"]
ONE_HOT_COLS = ['type_1']
LABEL_ENCODE_COLS = ['mailbox', 'innso_typology_ticket_', 'type_2', 'type_3','type_4']
FREQUENCY_ENCODE_COLS = []
NORMALIZE_ENCODE_COLS = []
STANDARDIZE_ENCODE_COLS = []


class Transform(ABC):
    @abstractmethod
    def transform(self, df):
        pass


class StringTransform(Transform):
    @abstractmethod
    def transform(self, df):
        pass

class NumericTransform(Transform):
    @abstractmethod
    def transform(self, df):
        pass

class OnehotEncode(StringTransform):
    def transform(self, df):
        if len(ONE_HOT_COLS) == 0:
            return df

        for col in ONE_HOT_COLS:
            df = pd.get_dummies(df, columns=[col])

        return df

class LabelEncode(StringTransform):
    def transform(self, df):
        if len(LABEL_ENCODE_COLS) == 0:
            return df

        for col in LABEL_ENCODE_COLS:
            df[col] = df[col].astype("category").cat.codes

        return df


class FrequencyEncode(StringTransform):
    def transform(self, df):
        if len(FREQUENCY_ENCODE_COLS) == 0:
            return df

        for col in FREQUENCY_ENCODE_COLS:
            df[col] = df[col].map(df[col].value_counts())

        return df


class WordEmbeddings(StringTransform):
    def transform(self, df):
        if len(WORD_EMBEDDING_COLS) == 0:
            return df

        df = df.reset_index(drop=True)
        cols_for_wrd_embedding = WORD_EMBEDDING_COLS
        dfs_to_concat = [df.drop(cols_for_wrd_embedding, axis=1)]

        for col in cols_for_wrd_embedding:
            tfidf = TfidfVectorizer(max_features=20)
            X = tfidf.fit_transform(df[col])
            tfidf_df = pd.DataFrame(
                X.toarray(),
                columns=[f"{col}_{word}" for word in tfidf.get_feature_names_out()]
            )
            dfs_to_concat.append(tfidf_df)

        df = pd.concat(dfs_to_concat, axis=1)
        return df


class Normalize(NumericTransform):
    def transform(self, df):
        if len(NORMALIZE_ENCODE_COLS) == 0:
            return df

        scaler = MinMaxScaler()
        df[NORMALIZE_ENCODE_COLS] = scaler.fit_transform(df[NORMALIZE_ENCODE_COLS])

        return df


class Standardize(NumericTransform):
    def transform(self, df):

        if len(STANDARDIZE_ENCODE_COLS) == 0:
            return df

        scaler = StandardScaler()
        df[STANDARDIZE_ENCODE_COLS] = scaler.fit_transform(df[STANDARDIZE_ENCODE_COLS])

        return df
