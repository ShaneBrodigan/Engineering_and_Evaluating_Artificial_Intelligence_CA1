from StringTransform import OnehotEncode, LabelEncode, WordEmbeddings
from NumericTransform import Normalize, Standardize

def process_data(data):
    one_hot = OnehotEncode()
    one_hot.transform(data)

    label_encode = LabelEncode()
    label_encode.transform(data)

    word_embeddings = WordEmbeddings()
    word_embeddings.transform(data)

    normalize = Normalize()
    normalize.transform(data)

    standardize = Standardize()
    standardize.transform(data)