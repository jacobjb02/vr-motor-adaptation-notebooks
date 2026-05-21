import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
        'experiment', 'ppid_full', 'speed_label', 'target_x_label', 'target_position_x_cm', 'target_position_z_cm', 'trial_num', 'phase', 'phase_trial_num', 'phase_target_trial_num', 'trial_num_target', 'global_cycle_num', 'cycle_target_num','cycle_TargetxPhase_num', 'launch_angle', 'launch_deviation', 'launch_Speed','distance_from_target', 'ball_dist_to_center_cm', 'signed_euclidean_cm', 'lateral_error_x_cm', 'depth_error_z_cm', 'target_hit', 'water_speed_binary', 'water_speed_m_s', 'sign_label','set_order', 'min_pos_from_target_x_cm', 'min_pos_from_target_z_cm','target_angle_90', 'ball_pos_x', 'ball_pos_z','final_ball_pos_x'
            ]

    # make copied subset of original df 
    data_subset = data[cols].copy()

    # modify column dtypes
    data_subset = data_subset.astype({
                                        'experiment': 'category',
                                        'water_speed_binary': 'int8',
                                        'ppid_full': 'category',
                                        'phase': 'category',
                                        'speed_label': 'category',
                                        'ball_pos_x': 'string',
                                        'ball_pos_z': 'string',
                                        'distance_from_target': 'float64'
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



def remove_baseline_outlier_trials_threshold(data,
                                   y_col,
                                   trial_col,
                                   phase_col='phase',
                                   baseline_string='baseline',
                                   error_threshold=50.0):

    # Create boolean masks for phase and threshold
    is_baseline = data[phase_col] == baseline_string
    is_outlier = np.abs(data[y_col]) >= error_threshold

    # Define the exact rows to drop (must be BOTH baseline AND an outlier)
    drop_mask = is_baseline & is_outlier
    indices_to_drop = data[drop_mask].index

    # Extract the specific trial numbers for logging before removal
    removed_trials = data.loc[drop_mask, trial_col].tolist()
    outlier_count = len(removed_trials)

    if outlier_count > 0:
        print(f"--- Baseline Trial Outlier Removal ({y_col}) ---")
        print(f"Trials exceeding absolute threshold of {error_threshold}: {outlier_count}")

    # Return the cleaned dataframe 
    cleaned_data = data.drop(index=indices_to_drop).copy()    

    remaining_baseline_max = cleaned_data.loc[cleaned_data[phase_col] == baseline_string, y_col].abs().max()
    print(f"Max absolute error remaining in baseline: {remaining_baseline_max}")
    
    return cleaned_data


def remove_baseline_outlier_trials_threshold_sd(data,
                                   y_col,
                                   trial_col,
                                   phase_col='phase',
                                   baseline_string='baseline',
                                   sd_error_threshold=2.0):
    
    # Isolate baseline data to calculate distribution statistics
    baseline_series = data.loc[data[phase_col] == baseline_string, y_col]
    
    if baseline_series.empty:
        print(f"Warning: No baseline data found for {y_col}. Returning original data.")
        return data

    baseline_mean = baseline_series.mean()
    baseline_std = baseline_series.std()
    
    # Calculate absolute threshold value
    abs_threshold_val = sd_error_threshold * baseline_std

    # Define masks
    is_baseline = data[phase_col] == baseline_string
    # Deviation from baseline mean exceeds the SD-based threshold
    is_outlier = np.abs(data[y_col] - baseline_mean) > abs_threshold_val

    # Identify indices to drop (Baseline AND Outlier)
    drop_mask = is_baseline & is_outlier
    indices_to_drop = data[drop_mask].index

    removed_trials = data.loc[drop_mask, trial_col].tolist()
    outlier_count = len(removed_trials)

    trial_count = len(baseline_series)

    # Logging
    print(f"--- Baseline Trial Outlier Removal ({y_col}) ---")
    print(f"Baseline Mean: {baseline_mean:.4f}, SD: {baseline_std:.4f}")
    print(f"SD Threshold: {sd_error_threshold} ({abs_threshold_val:.4f} units)")
    print(f"Trials removed: {outlier_count} / {trial_count} total baseline trials")

    # Return cleaned dataframe
    cleaned_data = data.drop(index=indices_to_drop).copy()
    
    return cleaned_data


def remove_outliers_per_trial(data, y_col, trial_col, water_col, target_col, phase_col, sd_error_threshold=3.0):

    # Calculate mean and sd FOR EACH TRIAL across the dataset
    trial_means = data.groupby([trial_col, water_col, target_col], observed=True)[y_col].transform('mean')
    trial_sds = data.groupby([trial_col, water_col, target_col], observed=True)[y_col].transform('std')
    
    # Calculate absolute deviation from the trial-specific mean
    deviation = np.abs(data[y_col] - trial_means)
    
    # Define mask: True if deviation exceeds the trial's specific threshold
    is_outlier = deviation > (sd_error_threshold * trial_sds)

    # check for plotting
    data['is_outlier'] = deviation > (sd_error_threshold * trial_sds)

    outlier_summary = (
        data[data['is_outlier']]
        .groupby([phase_col, trial_col], observed=True)
        .size()
        .reset_index(name='removed_count')
    )

    print(outlier_summary)
    
    # Logging
    outlier_count = is_outlier.sum()
    total_trials = len(data)
    print(f"--- Trial-by-Trial Outlier Removal ({y_col}) ---")
    print(f"SD Threshold: {sd_error_threshold}")
    print(f"Trials removed: {outlier_count} / {total_trials} total rows")
    
    # Return cleaned dataframe
    cleaned_data = data[~is_outlier].copy()

    # show where outliers
    plt.scatter(data['trial_num'], data['is_outlier'])
    plt.show()
    
    return cleaned_data
