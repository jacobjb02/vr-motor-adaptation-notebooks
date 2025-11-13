import numpy as np
import pandas as pd


def build_design_matrix(
                df,
                x_cols,
                y_col,
                n_lag=1,
                group_cols=None, 
                x_dummys=None,
                interactions=None,

            ):

    # Copy to avoid modifying original df
    df2 = df.copy()

    # Create lagged DV
    if group_cols is None:
            lag_col = f"{y_col}_lag{n_lag}"
            df2[lag_col] = df2[y_col].shift(n_lag)
    else:
        lag_col = f"{y_col}_lag{n_lag}"
        df2[lag_col] = df2.groupby(group_cols)[y_col].shift(n_lag)
            
        # Drop NaNs
    df2 = df2.dropna().reset_index(drop=True)

    # Create dummies
    dummy_list = []

    if x_dummys is not None:
        for col in x_dummys:
            dum = pd.get_dummies(df2[col], prefix=col[:4], drop_first=True)
            dummy_list.append(dum)
        dummies_df = pd.concat(dummy_list, axis=1)
    else:
        dummies_df is None


    # Add interactions
    interactions_list = []
    
    if interactions is not None:
        for v1, v2 in interactions:
            new_col = f"{v1}_x_{v2}"
            ints = df2[v1] * df2[v2]
            interactions_list.append(ints.rename(new_col))
        interactions_df = pd.concat(interactions_list, axis=1)
    else:
        interactions_df is None
            
    # Build X
    X_base = df2[x_cols + [lag_col]]

    if dummies_df is not None:
        X_base = pd.concat([X_base, dummies_df], axis=1)
    else:
        X_base = X_base

    if interactions_df is not None:
        X_base = pd.concat([X_base, interactions_df], axis=1)
    else:
        X_base = X_base


    # Add intercept
    X_sm = sm.add_constant(X_base)
    X_sm = X_sm.apply(pd.to_numeric, errors="raise")

    # Outcome
    y = df2[y_col]
    y = pd.to_numeric(y, errors="raise")


    return X_sm, y