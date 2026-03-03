def rename_headers(df):

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.strip()

    return df

def unnamed_col_handling(df):



    cols = df.columns
    unnamed_cols = []
    new_list = []

    for col in cols:
        uniques = df[col].unique()
        if col.startswith('unnamed'):
            unnamed_cols.append(col)

    unnamed_df = df[unnamed_cols]

    for col in unnamed_df.columns:
        unnamed_df_unique = unnamed_df[col].unique()
        if unnamed_df_unique.i:
            new_list.append(col)


    return new_list

def merge_dfs(df1, df2):
    # Merge datasets
    pass

class NaHandler():
    def dropna(self):
        # drop nulls
        pass
