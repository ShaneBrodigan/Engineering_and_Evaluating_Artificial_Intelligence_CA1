from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier,HistGradientBoostingClassifier,
                              VotingClassifier)
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd

class Model(ABC):

    def __init__(self):
        self.y_pred = None

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class Sklearn(Model):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def report(self, X_test, y_test):
        """Generated and printed a classification report with precision, recall and f1"""
        y_pred = self.predict(X_test)
        print(f"\n--- {self.__class__.__name__} Classification Report ---")
        print(classification_report(y_test, y_pred))

    def evaluate(self, X_test, y_test):
        """Printed the accuracy score for the model"""
        score = self.model.score(X_test, y_test)
        print(f"{self.__class__.__name__} Accuracy: {score:.2%}")

    def show_confusion_matrix(self, X_test, y_test):
        """Printed a formatted confusion matrix to the console"""
        y_pred = self.predict(X_test)
        labels = sorted(set(y_test) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"CONFUSION MATRIX")
        print(f"{' ' * 15} PREDICTED")
        print(f"ACTUAL {' ' * 8}" + "  ".join([str(l).ljust(5) for l in labels]))

        for i, label in enumerate(labels):
            row_values = "  ".join([str(val).ljust(5) for val in cm[i]])
            print(f"{str(label).ljust(14)} {row_values}")

        print(f"{'=' * 50}\n")

    def get_f1_score(self, X_test, y_test, average='weighted'):
        """Returned the F1 score for the most recent predictions"""
        y_pred = self.y_pred

        score = f1_score(y_test, y_pred, average=average)
        return score


class TensorFlow(Model):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def report(self, X_test, y_test):
        """Generated and printed a classification report for the TensorFlow model"""
        y_pred = self.predict(X_test)
        print(f"\n--- {self.__class__.__name__} Classification Report ---")
        print(classification_report(y_test, y_pred))

    def evaluate(self, X_test, y_test):
        """Printed the accuracy score for the TensorFlow model"""
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"{self.__class__.__name__} Accuracy: {acc:.2%}")

    def show_confusion_matrix(self, X_test, y_test):
        """Printed a formatted confusion matrix to the console"""
        y_pred = self.predict(X_test)

        labels = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"CONFUSION MATRIX")
        print(f"{' ' * 15} PREDICTED")
        print(f"ACTUAL {' ' * 8}" + "  ".join([str(l).ljust(5) for l in labels]))

        for i, label in enumerate(labels):
            row_values = "  ".join([str(val).ljust(5) for val in cm[i]])
            print(f"{str(label).ljust(14)} {row_values}")

        print(f"{'=' * 50}\n")

    def get_f1_score(self, X_test, y_test, average='weighted'):
        """Returned the F1 score for the most recent predictions"""
        y_pred = self.y_pred

        score = f1_score(y_test, y_pred, average=average)
        return score

class RandomForest(Sklearn):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X_train, y_train):
        """Trained the RandomForest classifier"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Returned class predictions for the given test set"""
        y_pred = self.model.predict(X_test)

        return y_pred

class AdaBoost(Sklearn):
    def __init__(self, **kwargs):
        self.model = AdaBoostClassifier(**kwargs)

    def fit(self, X_train, y_train):
        """Trained the AdaBoost classifier"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Returned class predictions for the given test set"""
        y_pred = self.model.predict(X_test)

        return y_pred


class ExtraTrees(Sklearn):
    def __init__(self, **kwargs):
        self.model = ExtraTreesClassifier(**kwargs)

    def fit(self, X_train, y_train):
        """Trained the ExtraTrees classifier"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Returned class predictions for the given test set"""
        y_pred = self.model.predict(X_test)

        return y_pred


class HistGradient(Sklearn):
    def __init__(self, **kwargs):
        self.model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X_train, y_train):
        """Trained the HistGradientBoosting classifier"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Returned class predictions for the given test set"""
        y_pred = self.model.predict(X_test)

        return y_pred


class Voting(Sklearn):
    def __init__(self, estimators, voting='hard', **kwargs):
        """Built a VotingClassifier from the provided estimator wrappers"""
        named_estimators = [(name, wrapper.model) for name, wrapper in estimators]

        self.model = VotingClassifier(estimators=named_estimators, voting=voting, **kwargs)

    def fit(self, X_train, y_train):
        """Trained the Voting ensemble classifier"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Returned class predictions for the given test set"""
        y_pred = self.model.predict(X_test)

        return y_pred


class NeuralNetwork(TensorFlow):
    def __init__(self, input_dim, num_classes, hidden_layers=[64, 32], dropout_rate=0.2):
        """Built and compiled a sequential Keras model with configurable hidden layers and dropout"""
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model

    def fit(self, X_train, y_train, epochs=50, batch_size=32):
        """Trained the neural network for the specified number of epochs"""
        validation_split = 0.1 if len(X_train) >= 10 else 0.0
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=validation_split
        )
        print(f"Neural Network trained for {epochs} epochs.")

    def predict(self, X_test):
        """Returned class predictions as integer indices via argmax"""
        predictions = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions, axis=1)

    def evaluate(self, X_test, y_test):
        """Printed the accuracy and loss from model evaluation"""
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Neural Network Accuracy: {acc:.2%}")

class ShallowNeuralNetwork(NeuralNetwork):
    pass

class DeepNeuralNetwork(NeuralNetwork):
    pass