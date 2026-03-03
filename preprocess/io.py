import pandas as pd

class Reader:

    def read_in(self, file_path):
        df = pd.read_csv(file_path)
        return df


"""
class Writer:
    def write_out(self, DataFrame):
        # write out dataframe
        d = 2
"""