import pandas as pd
import numpy as np



def add_cycles(data,
               n_trials,
               ppid_col,
               target_col,
               global_trial_col='trial_num',
               inc_phase=True,
               phase_str='phase'
              ):

    grouping_vars = [target_col, ppid_col]
    
    df = data.copy()
    global_counts = df.groupby(ppid_col, observed=True).cumcount()
    group_counts = df.groupby(grouping_vars, observed=True).cumcount()

    df['global_cycle_num'] = global_counts // n_trials + 1
    df['cycle_target_num'] = group_counts // n_trials + 1

    # block iteration within phase
    if inc_phase == True:
        grouping_vars = grouping_vars + [phase_str]
        counts = df.groupby(grouping_vars, observed=True).cumcount()
        df['cycle_TargetxPhase_num'] = counts // n_trials + 1
        
    
    return df