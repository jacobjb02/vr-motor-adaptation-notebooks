import numpy as np
import pandas as pd



def extract_columns_at_key_trials(data,
                                  trial_col,
                                  num_trials,
                                  target_col,
                                  phase_col,
                                  phase_str,
                                  trials_to_extract,
                                  launch_speed_col = 'launch_Speed',
                                  launch_dev_col = 'launch_deviation',
                                  show_parameter_changes = True,
                                  groupby_cols = ['ppid_full','target_x_label','timepoints']
                                 ):


    data_copy = data.copy()

    # filter data to selected phase
    phase_df = data_copy[data_copy[phase_col] == phase_str]

    # apply second filter to list of key trials
    flat_trials = np.ravel(trials_to_extract) # since we are using array we flatten first
    phase_df_trials = phase_df[phase_df[trial_col].isin(flat_trials)]

    # # make a new column that labels time point
    trials = [
        phase_df_trials[trial_col].isin(trials_to_extract[0]),
        phase_df_trials[trial_col].isin(trials_to_extract[1]),
        phase_df_trials[trial_col].isin(trials_to_extract[2])
    ]

    # name timepoints
    timepoints = ['early', 'mid', 'late']
    # make new column & label trials as timepoints
    phase_df_trials['timepoints'] = np.select(trials, timepoints)

    if show_parameter_changes:

        # groupby specified columns and calculate the mean
        df_agg = phase_df_trials.groupby(groupby_cols, as_index=False)[
            [launch_dev_col, launch_speed_col]
        ].mean()

        # rename the columns to show they are averages
        df_agg = df_agg.rename(columns={
            launch_dev_col: 'mean_dev',
            launch_speed_col: 'mean_speed'
        })

        # wide format
        df_wide = df_agg.pivot(
            index=['ppid_full', 'target_x_label'], 
            columns='timepoints', 
            values=['mean_dev', 'mean_speed']
        )
        df_wide.columns = [f"{val}_{time}" for val, time in df_wide.columns]
        df_wide = df_wide.reset_index()
        
        # calculate deltas
        df_wide['dev_mid_early'] = df_wide['mean_dev_mid'] - df_wide['mean_dev_early']
        df_wide['dev_late_mid'] = df_wide['mean_dev_late'] - df_wide['mean_dev_mid']
        df_wide['dev_late_early'] = df_wide['mean_dev_late'] - df_wide['mean_dev_early']
        
        df_wide['speed_mid_early'] = df_wide['mean_speed_mid'] - df_wide['mean_speed_early']
        df_wide['speed_late_mid'] = df_wide['mean_speed_late'] - df_wide['mean_speed_mid']
        df_wide['speed_late_early'] = df_wide['mean_speed_late'] - df_wide['mean_speed_early']
        
        # melt back to long format
        # extract just the delta columns and melt them so transition becomes a categorical 
        df_plot = pd.wide_to_long(
            df_wide, 
            stubnames=['dev', 'speed'], 
            i=['ppid_full', 'target_x_label'], 
            j='transition', 
            sep='_', 
            suffix=r'(mid_early|late_mid|late_early)'
        ).reset_index()
        
        
        display(df_plot.head(50))

    return df_plot