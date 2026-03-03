from abc import abstractmethod
import Transform

class NumericTransform(Transform):
    @abstractmethod
    def transform(self, data: str):
        pass


class Normalize(NumericTransform):
    def transform(self, data: str):
        # Example placeholder implementation
        return data


class Standardize(NumericTransform):
    def transform(self, data: str):
        # Example placeholder implementation
        return data