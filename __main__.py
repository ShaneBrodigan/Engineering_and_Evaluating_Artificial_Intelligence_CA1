import pandas as pd
from tensorflow.python.ops.parallel_for.pfor import WrappedTensor

from preprocess.inputoutput import Reader, Writer
import numpy as np
import preprocess.datacleaning as dc
from config import Config
from preprocess import feature_engineering
from model.model import RandomForest, AdaBoost, ExtraTrees, HistGradient, SGDModel, Voting, NeuralNetwork
from modelling.modelling import Modelling

def main():
    config = Config()
    reader = Reader()

    # Read in Csv files.
    df_1 = reader.read_in('./datasets/AppGallery.csv')
    df_2 = reader.read_in('./datasets/Purchasing.csv')

    # Renamed headers and dropped unnamed cols.
    df_1 = dc.rename_headers(df_1)
    df_1 = dc.drop_unnamed_cols(df_1)

    df_2 = dc.rename_headers(df_2)
    df_2 = dc.drop_unnamed_cols(df_2)

    # Merged dataframes.
    merged = dc.merge_dfs([df_1, df_2])

    # Converted all text in dataframe to lowercase.
    df = dc.col_rows_to_lowercase(merged)

    # Handled null values.
    na_handler = dc.NaHandler(essential_col_names=config.ESSENTIAL_COLS)
    df = na_handler.drop_na_rows(df)

    # Dropped useless columns.
    df = dc.drop_useless_cols(df)

    # Translated dataframe to english.
    # Also, cleaned dataframe to get rid of any noise prior creating word embeddings.

    #df = dc.translate_to_en(df)
    #writer = Writer()
    #writer.write_out(df, './translated_df.csv')
    df = reader.read_in('./translated_df.csv')

    # Feature Engineered Dataset.
    fe = feature_engineering.FeatureEngineering(df)
    df = fe.process_data()

    type_3 = df['type_3']
    type_4 = df['type_4']
    df = df.drop(columns=['type_3', 'type_4'])

    modelling = Modelling(df, target_col='type_2', test_size=0.3)

if __name__ == "__main__":
    main()