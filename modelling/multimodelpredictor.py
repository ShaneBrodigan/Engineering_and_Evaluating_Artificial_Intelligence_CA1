from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import Config
from model.model import (RandomForest, AdaBoost, ExtraTrees, HistGradient, SGDModel, Voting,
                         DeepNeuralNetwork, ShallowNeuralNetwork)
import numpy as np
import pandas as pd

class MultiModelPredictor:
    def __init__(self, df, target_col, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(df, target=target_col, test_size=test_size, random_state=42)
        self.best_model_pred = None
        self.best_model = None
        self.target_col = target_col
        self.best_f1_score = 0
        self.best_model_name = ""
        self.do_modelling()
        print(f"BEST OVERALL F1 Score: {self.best_f1_score} from {self.best_model_name}")

    def train_test_split(self, df, target, test_size=0.2, random_state=42):
        y = df[target]
        X = df.drop(target, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def do_modelling(self):
        c = Config()

        # Random Forest
        model = RandomForest(criterion='entropy', n_estimators=300)
        model.fit(self.X_train, self.y_train)
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        model.report(self.X_test, self.y_test)
        f1_score = model.get_f1_score(self.X_test, self.y_test, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, self.y_test)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)

        # AdaBoost
        model = AdaBoost(n_estimators=300, learning_rate=0.5)
        model.fit(self.X_train, self.y_train)
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        model.report(self.X_test, self.y_test)
        f1_score = model.get_f1_score(self.X_test, self.y_test, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, self.y_test)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)

        # ExtraTrees
        model = ExtraTrees(n_estimators=300, criterion='entropy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        model.report(self.X_test, self.y_test)
        f1_score = model.get_f1_score(self.X_test, self.y_test, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, self.y_test)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)

        # HistGradient
        model = HistGradient(max_iter=300, learning_rate=0.1)
        model.fit(self.X_train, self.y_train)
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        model.report(self.X_test, self.y_test)
        f1_score = model.get_f1_score(self.X_test, self.y_test, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, self.y_test)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)

        """
        # SDGModel
        model = SGDModel(loss='log_loss', max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        model.report(self.X_test, self.y_test)
        f1_score = model.get_f1_score(self.X_test, self.y_test, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, self.y_test)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)
        """

        # Voting Classifier pipeline
        rf = RandomForest(n_estimators=100, criterion='entropy')
        et = ExtraTrees(n_estimators=100, criterion='entropy')
        ada = AdaBoost(n_estimators=100)

        model = Voting(estimators=[
            ('rf_model', rf),
            ('et_model', et),
            ('ada_model', ada)], voting='hard')

        model.fit(self.X_train, self.y_train)
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        model.report(self.X_test, self.y_test)
        f1_score = model.get_f1_score(self.X_test, self.y_test, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, self.y_test)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)

        # Neural Network - Setup
        num_features = self.X_train.shape[1]
        total_unique = np.unique(np.concatenate([self.y_train, self.y_test]))
        num_classes = len(total_unique)

        le = LabelEncoder()
        le.fit(total_unique)
        y_train_enc = le.transform(self.y_train)
        y_test_enc = le.transform(self.y_test)

        # Neural Network - Shallow
        model = ShallowNeuralNetwork(input_dim=num_features, num_classes=num_classes, hidden_layers=[128, 64, 32])
        model.fit(self.X_train, y_train_enc, epochs=2) # REPLACE WITH 150!!!
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, y_test_enc)
        model.report(self.X_test, y_test_enc)
        f1_score = model.get_f1_score(self.X_test, y_test_enc, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, y_test_enc)
        model.y_pred = le.inverse_transform(model.y_pred)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)

        # Neural Network -  Deep
        model = DeepNeuralNetwork(input_dim=num_features, num_classes=num_classes, hidden_layers=[612, 256, 128, 32])
        model.fit(self.X_train, y_train_enc, epochs=2) # REPLACE WITH 150!!!
        model.y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, y_test_enc)
        model.report(self.X_test, y_test_enc)
        f1_score = model.get_f1_score(self.X_test, y_test_enc, average=c.SELECTED_F1_AVERAGE)
        print(f"f1_score: {f1_score}")
        model.show_confusion_matrix(self.X_test, y_test_enc)
        model.y_pred = le.inverse_transform(model.y_pred)
        self.f1_score_checker(model, average=c.SELECTED_F1_AVERAGE)

    def f1_score_checker(self, model, average='weighted'):
        f1_score = model.get_f1_score(self.X_test, self.y_test, average=average)

        if f1_score > self.best_f1_score:
            self.best_f1_score = f1_score
            self.best_model_pred = model.y_pred
            self.best_model_name = model.__class__.__name__
            self.best_model = model

    def predict_with_best(self, df):
        X_all = df.drop(self.target_col, axis=1)
        predictions = self.best_model.predict(X_all)
        new_df = X_all.copy()
        new_df[f'{self.target_col}_predictions'] = predictions
        return new_df