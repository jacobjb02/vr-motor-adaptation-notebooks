import numpy as np
import pandas as pd



def baseline_correct(data,
                     phase_col,
                     baseline_string,
                     y_col,
                     grouping_vars = ['ppid_full','target_x_label']
                    ):
    
    # filter data to baseline
    df_filtered = data[data[phase_col] == baseline_string]
    print("BASELINE:", df_filtered.shape)

    # group data by participant x target and calculate mean of error
    df_baseline = df_filtered.groupby(grouping_vars)[y_col].mean().reset_index()
    df_baseline = df_baseline.rename(columns={y_col: f"{y_col}_mean"})

    # Merge baseline means back into the main dataframe
    df_merged = pd.merge(data, df_baseline, on=grouping_vars, how='left', suffixes=('', '_bc'))

    df_merged[f"{y_col}_bc"] = df_merged[y_col] - df_merged[f"{y_col}_mean"]

    return(df_merged)
    


    

