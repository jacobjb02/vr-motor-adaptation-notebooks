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
        'experiment', 'ppid_full', 'speed_label', 'target_x_label', 'target_position_x_cm', 'target_position_z_cm', 'trial_num', 'phase', 'phase_target_trial_num', 'trial_num_target', 'launch_deviation', 'launch_Speed', 'ball_dist_to_center_cm', 'lateral_error_x_cm', 'depth_error_z_cm', 'target_hit', 'water_speed_binary', 'water_speed_m_s', 'sign_label','set_order', 'min_pos_from_target_x_cm', 'min_pos_from_target_z_cm','target_angle_90'
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


    

    