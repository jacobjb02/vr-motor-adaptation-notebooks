import numpy as np
import pandas as pd


def retain_baseline_washout(data,
                           phase_col,
                           experiment_type,
                           phases_to_keep = ['baseline','washout_1','washout_2'],
                           experiment_col = 'experiment',
                           water_col = 'water_speed_m_s'):

    
    data_baseline_washout = data[data[phase_col].isin(phases_to_keep)].copy()

    # Extract and join unique experiment strings
    unique_exps = data[experiment_col].unique().astype(str)
    exp_tags = "_".join(unique_exps[:2]) 
    if len(unique_exps) > 2:
        exp_tags += "_etc"
        
    # Extract unique water speeds, rounded to prevent floating point noise
    speeds = data[water_col].unique()
    speed_tags = "_".join([str(round(s, 1)).replace('-', 'neg').replace('.', '_') for s in np.sort(speeds)])
    
    # construct the combined filename
    file_name = f"baseline_washout_{exp_tags}_speed_{speed_tags}.csv"
    file_path = f"..\\data\\{experiment_type}\\{file_name}"
    
    data_baseline_washout.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")
    
    return data_baseline_washout