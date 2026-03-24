from enum import unique

import pandas as pd
from tensorflow.python.ops.parallel_for.pfor import WrappedTensor

from preprocess.feature_engineering import FeatureEngineering
from preprocess.inputoutput import Reader, Writer
import numpy as np
import preprocess.datacleaning as dc
from config import Config
from preprocess import feature_engineering
from model.model import RandomForest, AdaBoost, ExtraTrees, HistGradient, Voting, NeuralNetwork
from modelling.multimodelpredictor import MultiModelPredictor

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

    # Dropped useless columns which don't offer any training value to model
    df = dc.drop_useless_cols(df)

    # Translates dataframe to English and saves to CSV at root directory
    df = dc.translate_to_en(df)
    writer = Writer()
    writer.write_out(df, './translated_df.csv')
    df = reader.read_in('./translated_df.csv')

    # Feature Engineer Dataset by encoding
    fe = feature_engineering.FeatureEngineering(df)
    df, label_encoder = fe.process_data()   # Returns encoded dataframe for models and label_encoder for decoding later

    # Extracts type_3 and type_4 columns from dataframe prior to model training
    type_3 = df['type_3']
    type_4 = df['type_4']
    df = df.drop(columns=['type_3', 'type_4'])


    # ********* Chained Multi  *************

    # Type 2 Predictions
    predictor = MultiModelPredictor(df, target_col='type_2', test_size=0.3)
    type_2_predictions_df = predictor.predict_with_best(df)

    # Type 3 Predictions
    type_2_predictions_df['type_3'] = type_3
    na_handler = dc.NaHandler(essential_col_names=['type_3'])
    na_handled_df = na_handler.drop_na_rows(type_2_predictions_df)
    predictor = MultiModelPredictor(na_handled_df, target_col='type_3', test_size=0.3)
    type_3_predictions_df = predictor.predict_with_best(type_2_predictions_df)

    # Type 4 Predictions
    type_3_predictions_df['type_4'] = type_4
    na_handler = dc.NaHandler(essential_col_names=['type_4'])
    na_handled_df = na_handler.drop_na_rows(type_3_predictions_df)
    predictor = MultiModelPredictor(na_handled_df, target_col='type_4', test_size=0.3)
    type_4_predictions_df = predictor.predict_with_best(type_3_predictions_df)

    # Final encoded dataframe from chained multi method
    multi_chained_predictions = type_4_predictions_df


    # *********  Hierarchical  *************
    # Type 2 Predictions
    predictor = MultiModelPredictor(df, target_col='type_2', test_size=0.3)
    h_type_2_predictions_df = predictor.predict_with_best(df)

    # Type 3 Predictions
    h_type_2_predictions_df['type_3'] = type_3
    na_handler = dc.NaHandler(essential_col_names=['type_3'])
    h_na_handled_df = na_handler.drop_na_rows(h_type_2_predictions_df)

    fe = FeatureEngineering(h_na_handled_df)
    filtered_dfs_dict = fe.col_class_splitter('type_2')
    h_type_3_predictions_df = pd.DataFrame()

    for df in filtered_dfs_dict.values():
        # Checks that df >= 2 rows, if it is not try to train a model for it
        if len(df) < 2:
            h_type_3_predictions_df = dc.merge_dfs([h_type_3_predictions_df, df])
            continue
        predictor = MultiModelPredictor(df, target_col='type_3', test_size=0.3)
        predictions_df_for_cls = predictor.predict_with_best(df)
        h_type_3_predictions_df = dc.merge_dfs([h_type_3_predictions_df, predictions_df_for_cls])

    # Type 4 Predictions
    h_type_3_predictions_df['type_4'] = type_4
    na_handler = dc.NaHandler(essential_col_names=['type_4'])
    h_na_handled_df = na_handler.drop_na_rows(h_type_3_predictions_df)

    fe = FeatureEngineering(h_na_handled_df)
    filtered_dfs_dict = fe.col_class_splitter('type_3')
    h_type_4_predictions_df = pd.DataFrame()

    for df in filtered_dfs_dict.values():
        # Checks that df >= 2 rows, if it is not try to train a model for it
        if len(df) < 2:
            h_type_4_predictions_df = dc.merge_dfs([h_type_4_predictions_df, df])
            continue
        predictor = MultiModelPredictor(df, target_col='type_4', test_size=0.3)
        predictions_df_for_cls = predictor.predict_with_best(df)
        h_type_4_predictions_df = dc.merge_dfs([h_type_4_predictions_df, predictions_df_for_cls])

    # Final encoded dataframe from hierarchical method
    multi_hierarchical_predictions = h_type_4_predictions_df

    """
    print(h_type_4_predictions_df.info())
    print(h_type_4_predictions_df.shape)
    print(h_type_4_predictions_df.head(50))
    print(h_type_4_predictions_df.tail(50))
    print('\n')
    print('------------------------------------------------------------------------')
    print('\n')
    print(type_4_predictions_df.info())
    print(type_4_predictions_df.shape)
    print(type_4_predictions_df.head(50))
    print(type_4_predictions_df.tail(50))

    print('\n')
    print('------------------------------------------------------------------------')
    print('\n')
    """

    #Decoding Prediction columns and outputting to CSV
    multi_chained_unencoded_df = label_encoder.inverse_transform(multi_chained_predictions)
    hierarchical_unencoded_df = label_encoder.inverse_transform(multi_hierarchical_predictions)
    writer.write_out(multi_chained_unencoded_df, './multi_chained_predictions.csv')
    writer.write_out(hierarchical_unencoded_df, './hierarchical_predictions.csv')



if __name__ == "__main__":

    main()