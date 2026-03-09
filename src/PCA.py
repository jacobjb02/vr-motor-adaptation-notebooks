from sklearn.decomposition import PCA

import numpy as np
import pandas as pd



def compute_PCA_summary(data,
                features,
                group_cols,
                water_col,
                n_components
               ):

    
    # ensure x-axis is the first feature
    #assert features[0] == 'min_pos_from_target_x_cm', "First idex feature MUST belong to the X-dimension!"
    
    results = []

    data_current = data.loc[data[water_col] != 0.0]

    data_group = data_current.groupby(group_cols)
    # now we can group by each unique group and create a dataframe for each thanks to pandas
    for group_name, group_data in data_group:
        
        current_group_row = {} # empty dict. row for current group
        
        # grab and combine group_cols and group_name, index-by-index
        for col_name, value in zip(group_cols, group_name):
            current_group_row[col_name] = value 
            # value being our unique group value (e.g., water_speed {col_name} == -2.0 {value})
        
        X_subset = group_data[features]

        pca = PCA(n_components=n_components)
        pca.fit(X_subset)

        for i in range(n_components):

            pc_vector = pca.components_[i]

            pc_id = f"pc_vector_{i+1}"

            pc_x = pc_vector[0]
            pc_z = pc_vector[1]

            if pc_z < 0:
                pc_x, pc_z = -pc_x, -pc_z

            pc_deg = np.rad2deg(np.arctan2(pc_z, pc_x))
            pc_rad = np.arctan2(pc_z, pc_x)

            # store data
            current_group_row[f'{pc_id}_angle'] = pc_deg
            current_group_row[f'{pc_id}_pca_center_x'] = pca.mean_[0]
            current_group_row[f'{pc_id}_pca_center_z'] = pca.mean_[1]
            current_group_row[f'{pc_id}_c_X'] = pc_x
            current_group_row[f'{pc_id}_c_Z'] = pc_z
            
            current_group_row[f'{pc_id}_variance_ratio'] = pca.explained_variance_ratio_[i]
            
        # append stored data to results 
        results.append(current_group_row)

    return pd.DataFrame(results)



def compute_PCA_error(trial_df,
                   #PCA_df,
                   pc_deg,
                      target_x_col,
                      target_z_col,
                   features,
                   water_col
                     ):

    
    # remove any still-water trials
    data_current = trial_df.loc[trial_df[water_col] != 0.0].copy()
    
    # convert PC vector components into radians, making PC rad col
    data_current['pc_rad'] = np.arctan2(data_current['pc_vector_1_c_Z'], data_current['pc_vector_1_c_X'])

    # Create cos and sin now for when we unrotate the data later through vectorization
    cos_t = np.cos(-data_current['pc_rad'])
    sin_t = np.sin(-data_current['pc_rad'])

    # ensure x-axis is the first feature
    data_current['centered_x'] = data_current[features[0]] - data_current['pc_vector_1_pca_center_x']
    data_current['centered_z'] = data_current[features[1]] - data_current['pc_vector_1_pca_center_z'] 
    # unrotate the data using sin and cos on the centred data
    data_current['unrotated_x'] = (data_current['centered_x'] * cos_t) - (data_current['centered_z'] * sin_t)
    data_current['unrotated_z'] = (data_current['centered_x'] * sin_t) + (data_current['centered_z'] * cos_t)
    # apply same unrotation to targets
    data_current['tgt_centered_x'] = data_current[target_x_col] - data_current['pc_vector_1_pca_center_x']
    data_current['tgt_centered_z'] = data_current[target_z_col] - data_current['pc_vector_1_pca_center_z'] 

    data_current['tgt_unrotated_x'] = (data_current['tgt_centered_x'] * cos_t) - (data_current['tgt_centered_z'] * sin_t)
    data_current['tgt_unrotated_z'] = (data_current['tgt_centered_x'] * sin_t) + (data_current['tgt_centered_z'] * cos_t)
    # x and z errors
    data_current['PCA_error_X_cm'] = (data_current['unrotated_x'] - data_current['tgt_unrotated_x']) * -1
    data_current['PCA_error_Z_cm'] = (data_current['unrotated_z'] - data_current['tgt_unrotated_z']) 

    obtuse_mask = np.abs(data_current['pc_rad']) > (np.pi / 2)
    data_current.loc[obtuse_mask, 'PCA_error_Z_cm'] *= -1

    return data_current
       



    
                