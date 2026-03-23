from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier,HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.linear_model import SGDClassifier
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
        """Generates a detailed classification report including Precision, Recall, and F1-Score."""
        y_pred = self.predict(X_test)
        print(f"\n--- {self.__class__.__name__} Classification Report ---")
        print(classification_report(y_test, y_pred))

    def evaluate(self, X_test, y_test):
        """Standard accuracy evaluation for all Sklearn subclasses."""
        score = self.model.score(X_test, y_test)
        print(f"{self.__class__.__name__} Accuracy: {score:.2%}")

    def show_confusion_matrix(self, X_test, y_test):
        """Generates and prints a structured, readable confusion matrix."""
        y_pred = self.predict(X_test)
        labels = sorted(set(y_test) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Wrap in DataFrame for structure
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"CONFUSION MATRIX")
        print(f"{' ' * 15} PREDICTED")
        print(f"ACTUAL {' ' * 8}" + "  ".join([str(l).ljust(5) for l in labels]))

        # Iterate through the DataFrame to print rows with better spacing
        for i, label in enumerate(labels):
            row_values = "  ".join([str(val).ljust(5) for val in cm[i]])
            print(f"{str(label).ljust(14)} {row_values}")

        print(f"{'=' * 50}\n")

    def get_f1_score(self, X_test, y_test, average='weighted'):
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
        """Generates a detailed classification report for TensorFlow models."""
        y_pred = self.predict(X_test)
        print(f"\n--- {self.__class__.__name__} Classification Report ---")
        print(classification_report(y_test, y_pred))

    def evaluate(self, X_test, y_test):
        """Standard accuracy evaluation for TensorFlow models."""
        # Using the built-in Keras evaluate method
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"{self.__class__.__name__} Accuracy: {acc:.2%}")

    def show_confusion_matrix(self, X_test, y_test):
        """Generates and prints a structured confusion matrix to the console."""
        y_pred = self.predict(X_test)

        # In TF, we derive labels from the unique values in the ground truth
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
        y_pred = self.y_pred

        score = f1_score(y_test, y_pred, average=average)
        return score

class RandomForest(Sklearn):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

class AdaBoost(Sklearn):
    def __init__(self, **kwargs):
        self.model = AdaBoostClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred


class ExtraTrees(Sklearn):
    def __init__(self, **kwargs):
        self.model = ExtraTreesClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred


class HistGradient(Sklearn):
    def __init__(self, **kwargs):
        self.model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred


class SGDModel(Sklearn):
    def __init__(self, **kwargs):
        self.model = SGDClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred


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


class NeuralNetwork(TensorFlow):
    def __init__(self, input_dim, num_classes, hidden_layers=[64, 32], dropout_rate=0.2):
        model = tf.keras.Sequential()

        # Input Layer
        model.add(tf.keras.layers.Input(shape=(input_dim,)))

        # Hidden Layers
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        # Output Layer: Must have neurons equal to max(label) + 1
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model

    def fit(self, X_train, y_train, epochs=50, batch_size=32):
        # Increased visibility: changed verbose to 1 if you want to see training progress
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
        # Added verbose=0 to hide the progress bars during reporting
        predictions = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions, axis=1)

    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Neural Network Accuracy: {acc:.2%}")

class ShallowNeuralNetwork(NeuralNetwork):
    pass

class DeepNeuralNetwork(NeuralNetwork):
    pass