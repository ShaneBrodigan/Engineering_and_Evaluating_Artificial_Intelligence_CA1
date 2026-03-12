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

    """
    X_train, X_test, y_train, y_test = modelling.train_test_split(df, target='type_2', test_size=0.3)

    model = RandomForest(criterion='entropy', n_estimators=300)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.evaluate(X_test, y_test)
    model.report(X_test, y_test)
    model.show_confusion_matrix(X_test, y_test)

    model = AdaBoost(n_estimators=300, learning_rate=0.5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.evaluate(X_test, y_test)
    model.report(X_test, y_test)
    model.show_confusion_matrix(X_test, y_test)

    model = ExtraTrees(n_estimators=300, criterion='entropy', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.evaluate(X_test, y_test)
    model.report(X_test, y_test)
    model.show_confusion_matrix(X_test, y_test)

    model = HistGradient(max_iter=300, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.evaluate(X_test, y_test)
    model.report(X_test, y_test)
    model.show_confusion_matrix(X_test, y_test)

    model = SGDModel(loss='log_loss', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.evaluate(X_test, y_test)
    model.report(X_test, y_test)
    model.show_confusion_matrix(X_test, y_test)

    # Voting Classifier pipeline
    rf = RandomForest(n_estimators=100, criterion='entropy')
    et = ExtraTrees(n_estimators=100, criterion='entropy')
    ada = AdaBoost(n_estimators=100)

    ensemble = Voting(estimators=[
        ('rf_model', rf),
        ('et_model', et),
        ('ada_model', ada)], voting='hard')

    ensemble.fit(X_train, y_train)
    ensemble.evaluate(X_test, y_test)
    model.report(X_test, y_test)
    model.show_confusion_matrix(X_test, y_test)

    #Neural Network
    # 1. Determine shapes based on your preprocessed data
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))  # Count unique categories in the target

    # 2. Instantiate and run
    model = NeuralNetwork(input_dim=num_features, num_classes=num_classes, hidden_layers=[128,64,32])

    # We already have X_train, X_test, y_train, y_test from model.train_test_split
    model.fit(X_train, y_train, epochs=10)
    y_pred = model.predict(X_test)
    model.evaluate(X_test, y_test)
    model.report(X_test, y_test)
    model.show_confusion_matrix(X_test, y_test)
    """

if __name__ == "__main__":
    main()