import pandas as pd

class Reader:

    def read_in(self, file_path):
        """Read a CSV file from the given path and returned it as a dataframe"""
        df = pd.read_csv(file_path)
        return df


class Writer:
    def write_out(self, df, file_path):
        """Wrote the dataframe to a CSV file at the specified path"""
        try:
            df.to_csv(file_path, index=False)
            print(f"Successfully wrote file to: {file_path}")
        except Exception as e:
            print(f"Failed to write file to {file_path}: {e}")
