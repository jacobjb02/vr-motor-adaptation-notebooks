from sklearn.decomposition import PCA

import numpy as np
import pandas as pd



def compute_PCA_summary(data,
                features,
                group_cols,
                water_col,
                n_components
               ):
    
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
            current_group_row[f'{pc_id}_c_X'] = pc_x
            current_group_row[f'{pc_id}_c_Z'] = pc_z
            current_group_row[f'{pc_id}_variance_ratio'] = pca.explained_variance_ratio_[i]
            
        # append stored data to results 
        results.append(current_group_row)

    return pd.DataFrame(results)






def compute_PCA_df(data,
                features,
                group_cols,
                water_col,
                n_components
               ):
    
    data_current = data.loc[data[water_col] != 0.0]

    data_group = data_current.groupby(group_cols)
    # now we can group by each unique group and create a dataframe for each thanks to pandas
    for group_name, group_data in data_group:
        
        # grab and combine group_cols and group_name, index-by-index
        for col_name, value in zip(group_cols, group_name):
            current_group_row[col_name] = value 
            # value being our unique group value (e.g., water_speed {col_name} == -2.0 {value})
        
        X_subset = group_data[features]

        pca = PCA(n_components=n_components)
        pca.fit(X_subset)

        # build rotation matrix to unrotate  the data
        rotation = np.array([[np.cos(-pc_rad), -np.sin(-pc_rad)],
                             [np.sin(-pc_rad),  np.cos(-pc_rad)]])

        # move data to origin
        data_centered = data_group - pca.mean_
        # rotate data, making x axis the lateral error, and the z axis the depth error
        data_rotated = data_centered @ rotation.T
        # grab first x and z pos values for this group
        t_x = group_data['target_position_x'].iloc[0]
        t_z = group_data['target_position_z'].iloc[0]

        # since we rotating the error cloud position, we must align the target with the data again by rotating it too
        target_v = np.array([t_x, t_z]) - pca.mean_
        target_rotated = target_v @ rotation.T
    
        current_group_row['pca_center_x'] = pca.mean_[0]
        current_group_row['pca_center_z'] = pca.mean_[1]
    
        is_depth_dominant = abs(pc_vector[1]) > abs(pc_vector[0])
    
        if is_depth_dominant:
            # PC1 is Depth (Z), PC2 is Lateral (X)
            depth_idx = 0
            lateral_idx = 1
        else:
            # PC1 is Lateral (X), PC2 is Depth (Z) 
            depth_idx = 1
            lateral_idx = 0

        # Use the dynamic indices to grab the correct columns
        depth_error = data_rotated.iloc[:, depth_idx] - target_rotated[depth_idx]
        lateral_error = data_rotated.iloc[:, lateral_idx] - target_rotated[lateral_idx]





            

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
            data_rotated[f'{pc_id}_angle'] = pc_deg
            data_rotated[f'{pc_id}_c_X'] = pc_x
            data_rotated[f'{pc_id}_c_Z'] = pc_z
            data_rotated[f'{pc_id}_variance_ratio'] = pca.explained_variance_ratio_[i]

    return(data_rotated)




def apply_PCA_transformation(data,
                             group_cols,
                             features
                            ):


    # build rotation matrix to unrotate  the data
    rotation = np.array([[np.cos(-pc_rad), -np.sin(-pc_rad)],
                         [np.sin(-pc_rad),  np.cos(-pc_rad)]])

    data_features = data[features]
    
    data_group = data_features.groupby(group_cols)
    
    # move data to origin
    data_centered = data_group - pca.mean_
    # rotate data, making x axis the lateral error, and the z axis the depth error
    data_rotated = data_centered @ rotation.T
    # grab first x and z pos values for this group
    t_x = group_data['target_position_x'].iloc[0]
    t_z = group_data['target_position_z'].iloc[0]

    # since we rotating the error cloud position, we must align the target with the data again by rotating it too
    target_v = np.array([t_x, t_z]) - pca.mean_
    target_rotated = target_v @ rotation.T

    current_group_row['pca_center_x'] = pca.mean_[0]
    current_group_row['pca_center_z'] = pca.mean_[1]

    is_depth_dominant = abs(pc_vector[1]) > abs(pc_vector[0])

    if is_depth_dominant:
        # PC1 is Depth (Z), PC2 is Lateral (X)
        depth_idx = 0
        lateral_idx = 1
    else:
        # PC1 is Lateral (X), PC2 is Depth (Z) 
        depth_idx = 1
        lateral_idx = 0

        # Use the dynamic indices to grab the correct columns
        depth_error = data_rotated.iloc[:, depth_idx] - target_rotated[depth_idx]
        lateral_error = data_rotated.iloc[:, lateral_idx] - target_rotated[lateral_idx]

        current_group_row['error_lateral_mean_m'] = lateral_error.mean()
        current_group_row['error_depth_mean_m'] = depth_error.mean()

    
    
            


            
            
            

        
            
          
    
            
    
            

        

    

    

    



    
                