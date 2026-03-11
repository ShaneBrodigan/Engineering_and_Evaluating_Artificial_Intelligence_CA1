from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier,HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.linear_model import SGDClassifier

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
