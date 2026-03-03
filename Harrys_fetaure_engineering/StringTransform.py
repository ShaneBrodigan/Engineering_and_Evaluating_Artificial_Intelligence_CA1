from abc import abstractmethod
from Transform import Transform

class StringTransform(Transform):
    @abstractmethod
    def transform(self, data: str):
        pass


class OnehotEncode(StringTransform):
    def transform(self, data: str):
        print(f"one_hot: {data}")
        return data


class LabelEncode(StringTransform):
    def transform(self, data: str):
        print(f"label_encode: {data}")
        return data


class WordEmbeddings(StringTransform):
    def transform(self, data: str):
        print(f"word_embeddings: {data}")
        return data


