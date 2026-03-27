import numpy as np
import pandas as pd

def retain_baseline_washout(data,
                           phase_col,
                           experiment_type,
                           phases_to_keep = ['baseline','washout_1','washout_2'],
                           experiment_col = 'experiment',
                           water_col = 'water_speed_binary',
                           trial_col = 'phase_trial_num', 
                           target_col = 'target_x_label'):  

    data_baseline_washout = data[data[phase_col].isin(phases_to_keep)].copy()
    data_baseline_washout_still = data_baseline_washout[data_baseline_washout[water_col] < 1.0] 

    # Handle dual experiment type - limit to first 8 trials per target in washout phases
    if experiment_type == 'dual_generalization':
        washout_phases = [p for p in phases_to_keep if 'washout' in p]
        baseline_data = data_baseline_washout_still[~data_baseline_washout_still[phase_col].isin(washout_phases)]
        washout_data = data_baseline_washout_still[data_baseline_washout_still[phase_col].isin(washout_phases)]
        
        # Group by target and participant and take first 8 trials per target in washout
        #washout_data = washout_data.groupby([target_co, ppid_col]).apply(
         #   lambda x: x.nsmallest(8, trial_col) if trial_col in x.columns else x.head(8)
        #).reset_index(drop=True)
        
        data_baseline_washout_still = pd.concat([baseline_data, washout_data], ignore_index=True)
    
    # Extract and join unique experiment strings
    unique_exps = data_baseline_washout_still[experiment_col].unique().astype(str) 
    exp_tags = "_".join(unique_exps[:2]) 
    if len(unique_exps) > 2:
        exp_tags += "_etc"
        
    # Extract unique water speeds from filtered data
    speeds = data_baseline_washout_still[water_col].unique() 
    speed_tags = "_".join([str(round(s, 1)).replace('-', 'neg').replace('.', '_') for s in np.sort(speeds)])
    
    file_name = f"{phases_to_keep}_{exp_tags}_speed_{speed_tags}.csv"
    file_path = f"..\\data\\{experiment_type}\\{file_name}"
    
    data_baseline_washout_still.to_csv(file_path, index=False)  
    print(f"Saved: {file_path}")
    
    return data_baseline_washout_still 