from abc import ABC, abstractmethod

from numpy.matlib import empty
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from deep_translator import GoogleTranslator
from deep_translator.exceptions import NotValidLength
import spacy
import numpy as np

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
            codes = df[col].astype("category").cat.codes
            df[col] = codes.replace(-1, np.nan) # converting -1 values to nans, so nans can be handled.

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

class Translate(StringTransform):

    def translate_cols(self, df_to_translate):
        df_to_translate = df_to_translate.reset_index(drop=True)
        df_translate = df_to_translate.copy()

        for index, row in df_translate.iterrows():
            translated_ticket_summary = self.get_translation(row['ticket_summary'])
            translated_interaction_content = self.get_translation(row['interaction_content'])

            df_translate.at[index, 'ticket_summary'] = translated_ticket_summary
            df_translate.at[index, 'interaction_content'] = translated_interaction_content

        df_to_translate = df_to_translate.reset_index(drop=True)
        return df_translate

    def get_translation(self, content_to_translate):
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(content_to_translate)
            return translated
        except NotValidLength:
            print(
                f'Translation of {len(content_to_translate)} characters is too long for API, truncating to first 4900 words.')
            content_to_translate = content_to_translate[:4900]
            translated = GoogleTranslator(source='auto', target='en').translate(content_to_translate)
            return translated
        except Exception as ex:
            translated = content_to_translate
            print('[ERROR] During translation')
            return translated

    def clean_translated_cols(self, df_to_clean):
        df_to_clean = df_to_clean.reset_index(drop=True)
        nlp = spacy.load("en_core_web_sm")

        for index, row in df_to_clean.iterrows():
            # 1. Get the translated string
            raw_summary = row['ticket_summary']
            raw_content = row['interaction_content']

            # 2. CONVERT strings to spaCy Docs (This is what provides .is_stop)
            doc_summary = nlp(str(raw_summary))
            doc_content = nlp(str(raw_content))

            # 3. Clean the summary
            cleaned_summary_list = [token.lemma_.lower() for token in doc_summary
                                    if not token.is_stop and not token.is_punct and not token.is_space]

            # 4. Clean the content (This won't error now!)
            cleaned_content_list = [token.lemma_.lower() for token in doc_content
                                    if not token.is_stop and not token.is_punct and not token.is_space]

            # 5. Join the lists back into strings so they fit in the CSV/Dataframe
            summary_str = " ".join(cleaned_summary_list)
            content_str= " ".join(cleaned_content_list)

            df_to_clean.at[index, 'ticket_summary'] = summary_str
            df_to_clean.at[index, 'interaction_content'] = content_str

        # If spacy removes all words during cleaning it returns 'null' instead. This causes errors during word embedding
        df_to_clean['ticket_summary'] = df_to_clean['ticket_summary'].replace("null", "Empty")
        df_to_clean['ticket_summary'] = df_to_clean['ticket_summary'].fillna("Empty")
        df_to_clean['interaction_content'] = df_to_clean['interaction_content'].replace("null", "Empty")
        df_to_clean['interaction_content'] = df_to_clean['interaction_content'].fillna("Empty")

        return df_to_clean

    def transform(self, df):
        df = self.translate_cols(df)

        df = self.clean_translated_cols(df)

        return df


class Standardize(NumericTransform):
    def transform(self, df):

        if len(STANDARDIZE_ENCODE_COLS) == 0:
            return df

        scaler = StandardScaler()
        df[STANDARDIZE_ENCODE_COLS] = scaler.fit_transform(df[STANDARDIZE_ENCODE_COLS])

        return df
