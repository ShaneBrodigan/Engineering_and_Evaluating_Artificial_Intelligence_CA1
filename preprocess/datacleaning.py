def rename_headers(df):
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.strip()

    return df


def drop_unnamed_cols(df):
    unnamed_cols = df.columns[df.columns.str.startswith('unnamed')]
    no_unnamed_cols_df = df.drop(unnamed_cols, axis=1)

    return no_unnamed_cols_df


def merge_dfs(df1, df2):
    # Merge datasets
    pass


class NaHandler():
    def dropna(self):
        # drop nulls
        pass
