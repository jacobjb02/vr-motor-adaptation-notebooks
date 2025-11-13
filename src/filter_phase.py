import numpy as np
import pandas as pd


def filter_for_phase(
                        data,
                        phase_key,
                        block=False,
                        block_len=None,
                        inc_phase=True,
                        expected_n=43
                        ):
    
    # filter for select phase(s) only
    selected_phase = data[data['phase'].str.startswith(tuple(phase_key))].copy()
    
    # order within each (ppid, target) by trial_num_target
    selected_phase = selected_phase.sort_values(['ppid', 'target_x_label', 'phase', 'trial_num_target'])

    if inc_phase == True:
        # counter per participant x target and block across exposure trials
        selected_phase['phase_trial_target'] = selected_phase.groupby(['ppid', 'phase','target_x_label']).cumcount() + 1
    else:
        selected_phase['phase_trial_target'] = selected_phase.groupby(['ppid', 'target_x_label']).cumcount() + 1

    # makes block counter given number of trials per block
    if block == True:
        selected_phase['block'] = (selected_phase['phase_trial_target'] - 1) // block_len + 1

    # baseline (7) + washout (36)
    max_check = selected_phase.groupby(['ppid','target_x_label'])['phase_trial_target'].max()
    
    assert (max_check <= expected_n).all(), max_check
        
    selected_max_df = (
        selected_phase
        .groupby(['ppid', 'target_x_label'], observed=True)['phase_trial_target']
        .max()
        .reset_index()  # converts MultiIndex -> columns
        .rename(columns={'phase_trial_target': 'max_trial'}) 
                        )

    return (selected_phase, selected_max_df)
    
