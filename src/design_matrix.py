import numpy as np
import pandas as pd
import statsmodels.api as sm



def build_design_matrix(
                df,
                x_cols,
                y_col,
                ppid_col = 'ppid_full',
                add_constant=False,
                n_lag=[1,2,3],
                group_cols=None, 
                x_dummys=None,
                interactions=None,

            ):

    # Copy to avoid modifying original df
    df2 = df.copy()

    df2['ppid'] = df[ppid_col] 

    # Create lagged DV
    lag_cols = []

    for lag in n_lag:
        col_name = f"{y_col}_lag{lag}"
        if group_cols is None:
            df2[col_name] = df2[y_col].shift(lag)
        else:
            df2[col_name] = df2.groupby(group_cols)[y_col].shift(lag)
        
        lag_cols.append(col_name)
        
    # Drop rows where y or lag is missing
    df2 = df2.dropna(subset=[y_col] + lag_cols).reset_index(drop=True)


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
    X_base = df2[x_cols + lag_cols]

    if dummies_df is not None:
        X_base = pd.concat([X_base, dummies_df], axis=1)

    if interactions_df is not None:
        X_base = pd.concat([X_base, interactions_df], axis=1)

    # Add intercept
    if add_constant == True:
        X_sm = sm.add_constant(X_base)
        X_sm = X_sm.apply(pd.to_numeric, errors="raise")
    else:
        X_sm = X_base.copy()

    # Outcome
    y = pd.to_numeric(df2[y_col], errors="raise")

    # make df with lag columns
    lag_df = df2[lag_cols]   



    return X_sm.astype(float), y.astype(float), df2['ppid']
