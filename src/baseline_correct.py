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
    


    

def baseline_correct_locked(data, 
                           y_col='PCA_error_X_cm', 
                           phase_col='phase', 
                           baseline_string='training_1'):
    
    # 1. Calculate the mean error for each participant at each target in the NAIVE phase
    # This captures their initial 'constant error' under the perturbation
    df_baseline = data[data[phase_col] == baseline_string].groupby(['ppid_full', 'target_position_x_cm'])[y_col].mean().reset_index()
    df_baseline = df_baseline.rename(columns={y_col: f'{y_col}_baseline_mean'})

    # 2. Merge back
    df_merged = data.merge(df_baseline, on=['ppid_full', 'target_position_x_cm'], how='left')

    # 3. Correct: (Current Trial Error) - (Mean Naive Error)
    # This shows how much they IMPROVED relative to their own first 10-20 throws.
    df_merged[f'{y_col}_bc'] = df_merged[y_col] - df_merged[f'{y_col}_baseline_mean']
    
    return df_merged
