import pandas as pd
import numpy as np



def add_cycles(data,
               n_trials,
               grouping_vars,
               inc_phase=True,
               phase_str='phase'
              ):

    df = data.copy()
    counts = df.groupby(grouping_vars).cumcount()
    
    df['cycle_target_num'] = counts // n_trials + 1

    # block iteration within phase
    if inc_phase == True:
        grouping_vars = grouping_vars + ['phase']
        counts = df.groupby(grouping_vars).cumcount()
        df['cycle_TargetxPhase_num'] = counts // n_trials + 1
        
    
    return df