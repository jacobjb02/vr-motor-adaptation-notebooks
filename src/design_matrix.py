import numpy as np
import pandas as pd
import statsmodels.api as sm



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
    lag_col = f"{y_col}_lag{n_lag}"
    if group_cols is None:
        df2[lag_col] = df2[y_col].shift(n_lag)
    else:
        df2[lag_col] = df2.groupby(group_cols)[y_col].shift(n_lag)
    
    # Drop rows where y or lag is missing
    df2 = df2.dropna(subset=[y_col, lag_col]).reset_index(drop=True)


    # Create dummies
    dummies_df= None

    if x_dummys is not None:
            dummy_list = []
            for col in x_dummys:
                dum = pd.get_dummies(df2[col], prefix=col[:4], drop_first=True)
                dummy_list.append(dum)
            if dummy_list:
                dummies_df = pd.concat(dummy_list, axis=1)

    # Add interactions
    interactions_df = None
    
    if interactions is not None:
        interactions_list = []
        for v1, v2 in interactions:
            new_col = f"{v1}_x_{v2}"
            ints = df2[v1] * df2[v2]
            interactions_list.append(ints.rename(new_col)) 
        if interactions_list:
            interactions_df = pd.concat(interactions_list, axis=1)
            
    # Build X
    X_base = df2[x_cols + [lag_col]]

    if dummies_df is not None:
        X_base = pd.concat([X_base, dummies_df], axis=1)

    if interactions_df is not None:
        X_base = pd.concat([X_base, interactions_df], axis=1)

    # Add intercept
    X_sm = sm.add_constant(X_base)
    X_sm = X_sm.apply(pd.to_numeric, errors="raise")

    # Outcome
    y = pd.to_numeric(df2[y_col], errors="raise")

    # Lag series
    lag_series = pd.to_numeric(df2[lag_col], errors="raise")


    return X_sm, y, lag_series