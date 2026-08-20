from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting import plot_min_x_z



def compute_PCA_summary(data,
                features,
                group_cols,
                water_col,
                n_components
               ):

    results = []

    data_group = data.groupby(group_cols, observed=True)
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
        
    #print(results)

    return pd.DataFrame(results)



def compute_PCA_error(trial_df,
                   #PCA_df,
                   pc_deg,
                      target_x_col,
                      target_z_col,
                   features,
                      group_cols, 
                   water_col,
                      show_plots = False
                     ):

    # convert PC vector components into radians, making PC rad col
    trial_df['pc_rad'] = np.arctan2(trial_df['pc_vector_1_c_Z'], trial_df['pc_vector_1_c_X'])

    # Create cos and sin now for when we unrotate the data later through vectorization
    cos_t = np.cos(-trial_df['pc_rad'])
    sin_t = np.sin(-trial_df['pc_rad'])

    # ensure x-axis is the first feature, and center the data onto the targets. 
    trial_df['centered_x'] = trial_df[features[0]] - trial_df['pc_vector_1_pca_center_x']
    trial_df['centered_z'] = trial_df[features[1]] - trial_df['pc_vector_1_pca_center_z'] 
    
    # Check if target centered was correct
    assert np.isclose(trial_df.groupby(group_cols, observed=True)['centered_x'].mean(), 0, atol=1e-8).all(), "X not centered in one or more groups"
    assert np.isclose(trial_df.groupby(group_cols, observed=True)['centered_z'].mean(), 0, atol=1e-8).all(), "Z not centered in one or more groups"
            
    # unrotate the data using sin and cos on the centred data: Places all targets on the same plane for equal comparison
    trial_df['unrotated_x'] = (trial_df['centered_x'] * cos_t) - (trial_df['centered_z'] * sin_t)
    trial_df['unrotated_z'] = (trial_df['centered_x'] * sin_t) + (trial_df['centered_z'] * cos_t)

    # Diagnostic: Visualize unrotating target data-clouds    
    if show_plots == True:
        plot_min_x_z(
            data=trial_df,
            x_col='unrotated_x',
            y_col='unrotated_z',
            x_col_title='Unrotated X (cm)',
            y_col_title='Unrotated Z (cm)',
            c_col='target_x_label',
            r_col='water_speed_m_s',
            hue_col='water_speed_binary',
            x_lim=[-187.5, 187.5],
            y_lim=[-125, 250],
            # show_target=True,
            facet_height=4.0,
            facet_aspect=1.0
        )

    # apply same unrotation to targets
    trial_df['tgt_centered_x'] = trial_df[target_x_col] - trial_df['pc_vector_1_pca_center_x']
    trial_df['tgt_centered_z'] = trial_df[target_z_col] - trial_df['pc_vector_1_pca_center_z'] 

    trial_df['tgt_unrotated_x'] = (trial_df['tgt_centered_x'] * cos_t) - (trial_df['tgt_centered_z'] * sin_t)
    trial_df['tgt_unrotated_z'] = (trial_df['tgt_centered_x'] * sin_t) + (trial_df['tgt_centered_z'] * cos_t)
    
    # x and z orthogonal errors
    trial_df['PCA_error_X_cm'] = (trial_df['unrotated_x'] - trial_df['tgt_unrotated_x']) 
    trial_df['PCA_error_Z_cm'] = (trial_df['unrotated_z'] - trial_df['tgt_unrotated_z']) 

    
    # Diagnostic:   
    if show_plots == True:
        g = sns.relplot(
            data=trial_df, 
            x='PCA_error_X_cm', 
            y='PCA_error_Z_cm',
            col='target_x_label',  
            row='water_speed_binary',
            hue='target_hit',
            alpha=0.15,
            palette='viridis',            
            kind='scatter'
        )
        # Apply grid to all faceted subplots
        for ax in g.axes.flat:
            ax.grid(True)
        
        plt.show()



    # Abs PCA DV:
    # trial_df['abs_PCA_extent_error_cm'] = np.abs(trial_df['PCA_extent_error_cm'])


    return trial_df
       



    
                