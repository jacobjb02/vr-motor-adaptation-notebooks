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
        'ppid', 'trial_num', 'phase', 'trial_num_target', 'launch_deviation', 
        'launch_dev_z', 'launch_Speed', 'launch_Speed_z',
        'ball_dist_to_center_cm', 'target_hit', 'water_speed_binary', 'water_speed_m_s',
        'target_x_label','sign_label','set_order','ball_pos_x', 'ball_pos_z',
        'target_position_x', 'target_position_z','target_angle_90'
            ]

    # make copied subset of original df 
    data_subset = data[cols].copy()

    # modify column dtypes
    data_subset = data_subset.astype({
                                        'water_speed_binary': 'int8',
                                        'target_x_label': 'category',
                                        'ppid': 'category',
                                        'phase': 'category'
                                    })

    return(data_subset)


# Check if trials crossed Z-axis 70.0 cm threshold
def crossed_threshold(row, col, val):
    
    z_vals = [float(z) for z in str(row[col]).split('_') if z]
    
    return max(z_vals) >= val