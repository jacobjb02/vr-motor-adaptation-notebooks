import numpy as np
import pandas as pd

''' functions that label and define key phases of the experiment '''




def label_phases(n_phases = 6,
                phase_names = ['familiarization',
                               'baseline',
                               'training_1',
                               'washout_1',
                               'training_2',
                               'washout_2']
                ):

    assert len(phase_names) == n_phases, "Number of phases (n_phase) =/= number of specified phases (phase_names)"
    
    # create empty list of n_phases
    phase_array = np.empty(n_phases, dtype='object')

    # store named phases into phase_array
    phase_array[:] = phase_names
    
    return phase_array


def df_add_phases(
                        df, 
                       
                       phase_array,
                       
                       trial_lo = np.array([1,21,53,203,275,425]),

                       trial_hi = np.array([20,52,202,274,424,496])
    
                      ):


    for i in range(len(trial_lo)-1):
        assert trial_lo[i+1] - trial_hi[i] == 1, f"Gap or overlap between phase {i} and {i+1}"

    trial_lo, trial_hi = np.array(trial_lo), np.array(trial_hi)
    

    # assign empty rows to new phase col
    df['phase'] = pd.NA

    # loop through each phase, creating a mask to determine if trials fall in the set range
    for name, lo, hi in zip(phase_array, trial_lo, trial_hi):
        mask = (df['trial_num'] >= lo) & (df['trial_num'] <= hi)
        df.loc[mask, 'phase'] = name

    return df




    

def df_add_phases_old(df,
               
               baseline_min, baseline_max,

               training_1_min, training_1_max,
               washout_1_min, washout_1_max,
               
               training_2_min, training_2_max,
               washout_2_min, washout_2_max
    
              ):

    df['phase'] = pd.NA


    # make boolean masks for each phase:
    baseline_mask = (df['trial_num'] >= baseline_min) & (df['trial_num'] <= baseline_max)
    training_1_mask = (df['trial_num'] >= training_1_min) & (df['trial_num'] <= training_1_max)
    washout_1_mask = (df['trial_num'] >= washout_1_min) & (df['trial_num'] <= washout_1_max)
    training_2_mask = (df['trial_num'] >= training_2_min) & (df['trial_num'] <= training_2_max)
    washout_2_mask = (df['trial_num'] >= washout_2_min) & (df['trial_num'] <= washout_2_max)


    df.loc[baseline_mask, 'phase'] = 'baseline'
    
    df.loc[training_1_mask, 'phase'] = 'training_1'
    df.loc[washout_1_mask, 'phase'] = 'washout_1'

    df.loc[training_2_mask, 'phase'] = 'training_2'
    df.loc[washout_2_mask, 'phase'] = 'washout_2'

    return(df['phase'])

        

    

    




    

    