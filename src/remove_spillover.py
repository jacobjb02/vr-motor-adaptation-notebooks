import numpy as np
import pandas as pd






def fix_washout_trials(data,
                        group_cols = ['ppid_full', 'target_x_label', 'phase', 'block', 'water_speed_binary'],
                       ):

        sorted_data = data.sort_values(['ppid_full','target_x_label','phase','block','trial_num_target'])

        sorted_data['prev_speed'] = sorted_data['water_speed_binary'].shift(1)

        sorted_data['context_transition'] = sorted_data['water_speed_binary'] != sorted_data['prev_speed']
            
        # filter data when context_transition == True
        out = sorted_data[sorted_data['context_transition'] == False]
    
        return out



    

def remove_spillover(data,
                     group_cols=['block','water_speed_binary']
                    ):

    # count number of rows in each group
    group_counts = data.groupby(group_cols).size().rename('count').reset_index()

      # identify blocks with more than one water_speed_binary group
    valid_blocks = (group_counts.groupby('block').filter(lambda x: len(x) > 1))

    # label the small groups
    spillover_map = valid_blocks.loc[valid_blocks.groupby('block')['count'].idxmin()].rename(columns={'water_speed_binary': 'spillover_speed'})

    # since the last block contains no spillover, the function incorrectly labels the final block
    #spillover_map = spillover_map.drop(spillover_map.index[-1])

    # check if the original data's row = the spill_over data water_speed_binary during this block, if so, label
    data = data.merge(spillover_map[['block','spillover_speed']], on='block', how='left')

    # label spillover rows
    data['is_spillover'] = data['water_speed_binary'] == data['spillover_speed']

    return data


    


    



    