from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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