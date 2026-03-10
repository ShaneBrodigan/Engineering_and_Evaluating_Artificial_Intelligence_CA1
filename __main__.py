import pandas as pd
from tensorflow.python.ops.parallel_for.pfor import WrappedTensor

from preprocess.inputoutput import Reader, Writer
import numpy as np
import preprocess.datacleaning as dc
from config import Config
from preprocess import feature_engineering

def main():
    config = Config()

    reader = Reader()
    df_1 = reader.read_in('./datasets/AppGallery.csv')
    df_2 = reader.read_in('./datasets/Purchasing.csv')

    df_1 = dc.rename_headers(df_1)
    df_1 = dc.drop_unnamed_cols(df_1)

    df_2 = dc.rename_headers(df_2)
    df_2 = dc.drop_unnamed_cols(df_2)

    merged = dc.merge_dfs([df_1, df_2])

    df = dc.col_rows_to_lowercase(merged)

    na_handler = dc.NaHandler(essential_col_names=config.ESSENTIAL_COLS)
    df = na_handler.drop_na_rows(df)

    df = dc.drop_useless_cols(df)

    # df = dc.translate_to_en(df)
    # writer = Writer()
    # writer.write_out(df, './translated_df.csv')
    df = reader.read_in('./translated_df.csv')

    fe = feature_engineering.FeatureEngineering(df)
    df = fe.process_data()

    print(df.info())

    print("Testing...")

if __name__ == "__main__":
    main()