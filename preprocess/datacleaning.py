import pandas as pd


def rename_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.strip()

    return df


def col_rows_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    word_cols = df.select_dtypes(include='object').columns
    for col in word_cols:
        df[col] = df[col].str.lower()

    return df


def drop_unnamed_cols(df: pd.DataFrame) -> pd.DataFrame:
    unnamed_cols = df.columns[df.columns.str.startswith('unnamed')]
    no_unnamed_cols_df = df.drop(unnamed_cols, axis=1)

    return no_unnamed_cols_df


def merge_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged_df = pd.concat(dfs)
    no_dups_df = drop_true_duplicate_rows(merged_df)

    return no_dups_df


def drop_true_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    dups_removed = df.drop_duplicates(inplace=False)

    return dups_removed

def drop_useless_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(['ticket_id','interaction_id', 'interaction_date'], axis=1, inplace=True)

    return df

class NaHandler():
    essential_cols = []

    def __init__(self, essential_col_names: list) -> None:
        self.essential_cols = essential_col_names


    def drop_na_rows(self, df):
        cols = self.essential_cols

        for col in cols:
            df = df[~df[col].isna()]
        return df



