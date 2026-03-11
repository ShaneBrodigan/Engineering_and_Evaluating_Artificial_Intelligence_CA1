from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier,HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.linear_model import SGDClassifier
import tensorflow as tf

class Model(ABC):
    def train_test_split(self, df, target, test_size=0.2):
        y = df[target]
        X = df.drop(target, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        return X_train, X_test, y_train, y_test

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

class TensorFlow(Model):
    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

class RandomForest(Sklearn):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def evaluate(self, X_test, y_test):
        # Separate method for scoring
        score = self.model.score(X_test, y_test)
        print(f"RandomForest Accuracy: {score:.2%}")

class AdaBoost(Sklearn):
    def __init__(self, **kwargs):
        self.model = AdaBoostClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def evaluate(self, X_test, y_test):
        # Uses the built-in sklearn score method
        score = self.model.score(X_test, y_test)
        print(f"AdaBoost Accuracy: {score:.2%}")

class ExtraTrees(Sklearn):
    def __init__(self, **kwargs):
        self.model = ExtraTreesClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        print(f"Extra Trees Accuracy: {score:.2%}")

class HistGradient(Sklearn):
    def __init__(self, **kwargs):
        self.model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        print(f"HistGradientBoosting Accuracy: {score:.2%}")

class SGDModel(Sklearn):
    def __init__(self, **kwargs):
        self.model = SGDClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        print(f"SGDClassifier Accuracy: {score:.2%}")

class Voting(Sklearn):
    def __init__(self, estimators, voting='hard', **kwargs):
        """
        estimators: A list of (name, model_instance) tuples.
                    Example: [('rf', RandomForestClassifier()), ('et', ExtraTreesClassifier())]
        voting: 'hard' (majority vote) or 'soft' (weighted probabilities)
        """
        # We extract the actual sklearn model objects from your wrapper classes
        named_estimators = [(name, wrapper.model) for name, wrapper in estimators]

        self.model = VotingClassifier(estimators=named_estimators, voting=voting, **kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def evaluate(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        print(f"Voting Classifier Accuracy: {score:.2%}")


class NeuralNetwork(TensorFlow):
    def __init__(self, input_dim, num_classes, hidden_layers=[64, 32], dropout_rate=0.2):
        """
        input_dim: Number of features in X
        num_classes: Number of unique categories in y
        hidden_layers: List containing number of neurons per layer
        """
        model = tf.keras.Sequential()

        # Input Layer
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        # Hidden Layers
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))  # Prevents overfitting

        # Output Layer: Softmax scales outputs to probabilities that sum to 1
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model

    def fit(self, X_train, y_train, epochs=50, batch_size=32):
        # We use a validation split to monitor performance during training
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.1
        )

    def predict(self, X_test):
        # Returns probabilities for each class; we take the index of the highest
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Neural Network Accuracy: {acc:.2%}")