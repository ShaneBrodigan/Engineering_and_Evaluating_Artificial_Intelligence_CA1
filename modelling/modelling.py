from sklearn.model_selection import train_test_split
from model.model import RandomForest, AdaBoost, ExtraTrees, HistGradient, SGDModel, Voting, NeuralNetwork

class Modelling:
    def __init__(self, df, target_col, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(df, target=target_col, test_size=test_size)
        self.do_modelling()

    def train_test_split(self, df, target, test_size=0.2):
        y = df[target]
        X = df.drop(target, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        return X_train, X_test, y_train, y_test

    def do_modelling(self):

        # Random Forest
        model = RandomForest(criterion='entropy', n_estimators=300)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        model.evaluate(self.X_test, self.y_test)
        model.report(self.X_test, self.y_test)
        model.show_confusion_matrix(self.X_test, self.y_test)
