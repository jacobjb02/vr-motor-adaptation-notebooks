import pandas as pd


def filter_first_last_trials(data, 
                             phase_list, 
                             x_trials, 
                             phase_col='phase', 
                             trial_col='trial_num_target',
                             ppid_col='ppid_full',
                             target_col='target_x_label'):
    """
    Filters dataframe by phase(s) and returns first and last x trials per participant-target combination.
    """
    
    # Ensure phase_list is a list
    if isinstance(phase_list, str):
        phase_list = [phase_list]
    
    # Filter for selected phase(s)
    df_filtered = data[data[phase_col].isin(phase_list)].copy()
    
    # Sort by participant, target, and trial number
    df_filtered = df_filtered.sort_values([ppid_col, target_col, phase_col, trial_col])
    
    # Get first x trials per group
    first_x = df_filtered.groupby([ppid_col, target_col, phase_col], observed=True).head(x_trials)
    
    # Get last x trials per group
    last_x = df_filtered.groupby([ppid_col, target_col, phase_col], observed=True).tail(x_trials)
    
    # Combine and remove duplicates (in case groups have < 2x trials)
    result = pd.concat([first_x, last_x]).drop_duplicates().reset_index(drop=True)
    
    # Sort for clean output
    result = result.sort_values([ppid_col, target_col, phase_col, trial_col])
    
    return result
