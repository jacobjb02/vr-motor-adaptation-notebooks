"""
Load parquet file from R.
"""

import pandas as pd




def load_data(file_path, keep_col=None):
    """
    Load a parquet file into a pandas DataFrame.
    Drops rows with any NaN values, *except* in the specified keep_col.
    """

    df = pd.read_parquet(file_path)

    if keep_col and keep_col in df.columns:
        # mask of rows where all other columns are non-NaN
        non_keep_cols = [c for c in df.columns if c != keep_col]
        mask = df[non_keep_cols].notna().all(axis=1)

        # keep all rows that are fully valid in other columns
        df = df.loc[mask]
    else:
        df = df.dropna()

    return df




def load_data2(file_path):
    """
    Load a parquet file into a pandas DataFrame.
    Drops rows with any NaN values, *except* in the specified keep_col.
    """

    df = pd.read_parquet(file_path)

    return df


