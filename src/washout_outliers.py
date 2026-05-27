import pandas as pd


def label_outliers_washout(df, 
                           phase_col, 
                           water_state_col, 
                           threshold_col, 
                           threshold_val,
                           washout_keys = ['washout_1','washout_2']
                          ):

    df = df.copy()

    mask = (df[phase_col].isin(washout_keys)) & (df[water_state_col] == 0.0)

    df['is_outlier'] = False
    df.loc[mask, 'is_outlier'] = df.loc[mask, threshold_col] > threshold_val


    return df