import numpy as np
import pandas as pd



def combine_dfs(df_1, df_2, 
                cols_to_drop = []
               ):


    common_cols = df_1.columns.intersection(df_2.columns)

    df_c = pd.concat([df_1[common_cols], df_2[common_cols]], ignore_index = True)

    df_cf = df_c.drop(columns=cols_to_drop, errors='ignore')

    
    # make new x_col that makes a new trial counter for combined phases dfs
    if len(df_cf['phase'].unique()) > 1:
        df_cf['multi_phase_target_trial'] = df_cf.groupby(['ppid_full', 'target_x_label']).cumcount() + 1    

    return df_cf