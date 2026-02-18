"""
Adding important columns or investigating data values.
"""

import numpy as np
import pandas as pd


# function to extract the point along the ball trajectory when the ball's x position is closest to the target x position, and find the distance
def point_of_crossing_x(row):

    z_vals = [float(z) for z in row['ball_pos_z'].split('_')]
    x_vals = [float(x) for x in row['ball_pos_x'].split('_')]

    diffs_z = [abs(z - row['target_position_z']) for z in z_vals]
    idx = diffs_z.index(min(diffs_z))

    x_at_closest = x_vals[idx]

    return (x_at_closest - row['target_position_x']) * 100 # convert to cm


def get_sample_sizes(data, group_cols, ppid_col):
    return data.drop_duplicates(subset=[ppid_col] + (group_cols if isinstance(group_cols, list) else [group_cols])) \
               .groupby(group_cols).size()
    
# grab phase data and summarize
def summarize_phase(data, phase, ppid_col, error_col, group_col):

    # filter so we only have baseline data, make a copy
    phase_df = data.copy()
    phase_df = data[data['phase'] == phase]


    # filter trials given n
    #phase_df = phase_df[phase_df['trial_num_target'] >= trial_start]

    
    #print(base['ball_dist_to_center_cm'].min())
    
    #base = base.sort_values(['ppid','target_x_label','trial_num_target'])

    # summarize baseline data by ppid x target and obtain baseline errors
    phase_df_summary = (
        phase_df.groupby([ppid_col,group_col], as_index=False, observed=True).agg(
            mean_error_cm=(error_col,'mean'),
            mean_ball_launch_dev=('launch_deviation','mean'),
            mean_ball_launch_speed=('launch_Speed','mean')
        )
    )
    
    return phase_df_summary



# def add_training_status(data):

    #exp = data.query('trial_num_target > 13 and water_speed_binary == 1').copy()

   # exp.loc[exp



def order_targets(data,
                  order_array=np.array([1,0,2,3]),
                  order_col='target_x_label'):

    orig = np.unique(data[order_col])
    
    target_layout_order = np.array(order_array)
    
    new_order = orig[target_layout_order]
    
    # apply as categorical ordering
    data[order_col] = pd.Categorical(
        data[order_col],
        categories=new_order,
        ordered=True
    )

    print(data[order_col])

    return data[order_col]
    
