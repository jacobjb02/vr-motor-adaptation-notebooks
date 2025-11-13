import numpy as np
import pandas as pd


def filter_for_phase(
                        data,
                        phase_key,
                        block,
                        block_len
                        ):
    
    # filter for select phase only
    selected_phase = data[data['phase'].str.startswith(phase_key)]
    
    # order within each (ppid, target) by trial_num_target
    selected_phase = selected_phase.sort_values(['ppid', 'target_x_label', 'phase', 'trial_num_target'])

    # counter per participant x target and block across exposure trials
    selected_phase['phase_trial_target'] = selected_phase.groupby(['ppid', 'target_x_label']).cumcount() + 1

    if block == True:
        selected_phase.groupby(['pp'phase', 'phase_trial_target'])
    
    return selected_phase

