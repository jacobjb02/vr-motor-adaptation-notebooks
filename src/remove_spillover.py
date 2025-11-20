import numpy as np
import pandas as pd






def fix_washout_trials(data,
                        group_cols = ['ppid_full', 'target_x_label', 'phase', 'block', 'water_speed_binary'],
                        lastn_trials_to_keep=2
                       ):

        sorted_data = data.sort_values(['ppid_full','target_x_label','phase','block','trial_num_target'])

        out = sorted_data.groupby(group_cols).tail(lastn_trials_to_keep)

        return out



    

def remove_spillover(data,
                     group_cols=['block','water_speed_binary']
                    ):

    # count number of rows in each group
    group_counts = data.groupby(group_cols).size().rename('count').reset_index()

    # label the small groups
    spillover_map = group_counts.loc[group_counts.groupby('block')['count'].idxmin()].rename(columns={'water_speed_binary': 'spillover_speed'})

    # since the last block contains now spillover, the function incorrectly labels the final block
    spillover_map = spillover_map.drop(spillover_map.index[-1])

    # check if the original data's row = the spill_over data water_speed_binary during this block, if so, label
    data = data.merge(spillover_map, on='block', how='left')

    # label spillover rows
    data['is_spillover'] = data['water_speed_binary'] == data['spillover_speed']

    return data


    


    



    