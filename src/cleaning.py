import pandas as pd
import numpy as np

"""
Selecting, filtering, and cleaning data.
"""







def extract_key_columns(data):

    """
    Extracts selected columns from a larger DataFrame
    Include DataFrame as an argument
    """
    
    # selected columns (edit if needed)
    cols = [
        'experiment', 'ppid_full', 'speed_label', 'target_x_label', 'target_position_x_cm', 'target_position_z_cm', 'trial_num', 'phase', 'phase_target_trial_num', 'trial_num_target', 'global_cycle_num', 'cycle_target_num','cycle_TargetxPhase_num', 'launch_deviation', 'launch_Speed', 'ball_dist_to_center_cm', 'signed_euclidean_cm', 'lateral_error_x_cm', 'depth_error_z_cm', 'target_hit', 'water_speed_binary', 'water_speed_m_s', 'sign_label','set_order', 'min_pos_from_target_x_cm', 'min_pos_from_target_z_cm','target_angle_90'
            ]

    # make copied subset of original df 
    data_subset = data[cols].copy()

    # modify column dtypes
    data_subset = data_subset.astype({
                                        'experiment': 'category',
                                        'water_speed_binary': 'int8',
                                        'ppid_full': 'category',
                                        'phase': 'category',
                                        'speed_label': 'category'
                                    })

    return(data_subset)


# Check if trials crossed Z-axis 70.0 cm threshold
def crossed_threshold(row, col, val):
    
    z_vals = [float(z) for z in str(row[col]).split('_') if z]
    
    return max(z_vals) >= val



# attempts to floor min distance measure
def distance_target_ball_radii(data, 
                               y_col, 
                               target_radius=3.75, 
                               ball_radius=5.0,
                               epsilon=1e-6):

    radii_sum = target_radius + ball_radius
    corrected = data[y_col] - radii_sum
    
    # ensure non-negativity
    corrected = corrected.clip(lower=epsilon)
    
    data[f"{y_col}_radii_fixed"] = corrected
    return corrected


def flag_outlier_participants(data,
                              y_col,
                              phase_col='phase',
                              baseline_string='baseline',
                              target_col='target_x_label',
                              ppid_col='ppid_full',
                              sd_threshold=2):

    # Isolate baseline trials
    is_baseline = data[phase_col] == baseline_string
    df_base = data[is_baseline].copy()

    # Calculate each participant's average error per target
    # This reduces the data from trial-level to participant-target-level
    subj_means = df_base.groupby([ppid_col, target_col])[y_col].mean().reset_index(name='subj_target_mean')

    # Calculate the GROUP mean and GROUP SD for each target
    # Grouping ONLY by target_col evaluates the population distribution
    subj_means['group_target_mean'] = subj_means.groupby(target_col)['subj_target_mean'].transform('mean')
    subj_means['group_target_sd'] = subj_means.groupby(target_col)['subj_target_mean'].transform('std')

    # Identify outlier participants (inter-subject comparison)
    subj_means['is_participant_outlier'] = (
        np.abs(subj_means['subj_target_mean'] - subj_means['group_target_mean']) > 
        (subj_means['group_target_sd'] * sd_threshold)
    )

    # Extract the unique IDs of participants who failed the threshold on any target
    outlier_ppids = subj_means.loc[subj_means['is_participant_outlier'], ppid_col].unique()

    if len(outlier_ppids) > 0:
        print(f"--- Inter-Subject Outlier Detection ({y_col}) ---")
        print(f"Participants exceeding {sd_threshold} SD from group mean: {outlier_ppids.tolist()}")
        
        # Breakdown of which targets they failed on
        print("\nTarget breakdown for outliers:")
        print(subj_means[subj_means['is_participant_outlier']][[ppid_col, target_col, 'subj_target_mean', 'group_target_mean']])
        print("---------------------------------")
    else:
        print(f"No participant-level outliers detected for {y_col} at {sd_threshold} SD.")

    return outlier_ppids

    