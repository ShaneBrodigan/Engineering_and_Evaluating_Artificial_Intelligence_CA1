from abc import ABC, abstractmethod

def process():
    # ??? Requested by harry
    pass

class Transform(ABC):
    @abstractmethod
    def transform(self):
        pass

class StringTransformer:

    class WordEmbeddings:
        pass

    class OneHotEncoding:
        pass

    class LabelEncoding:
        pass

class NumberTransformer:

    class StandardScaling:
        pass

    class MinMaxScaling:
        pass

    class Normalization:
        pass