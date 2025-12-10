from sklearn.decomposition import PCA

import numpy as np
import pandas as pd



def compute_PCA(data,
                features,
                group_cols,
                water_col,
                n_components
               ):

    results = []

    # filter isnt working
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
        pcs = pca.fit_transform(X_subset)
        
        for i in range(n_components):

            pc_vector = pca.components_[i]

            pc_id = f"pc_vector_{i+1}"

            pc_x = pc_vector[0]
            pc_z = pc_vector[1]

            pc_deg = np.rad2deg(np.arctan2(pc_z, pc_x))

            current_group_row[f'{pc_id}_angle'] = pc_deg
            current_group_row[f'{pc_id}_c_X'] = pc_x
            current_group_row[f'{pc_id}_c_Z'] = pc_z
            current_group_row[f'{pc_id}_variance_ratio'] = pca.explained_variance_ratio_[i]

        results.append(current_group_row)

    return pd.DataFrame(results)
            


            
            
            

        
            
          
    
            
    
            

        

    

    

    



    
                