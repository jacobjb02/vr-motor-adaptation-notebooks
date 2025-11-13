"""
Adding important columns or investigating data values.
"""

import numpy as np


# function to extract the point along the ball trajectory when the ball's x position is closest to the target x position, and find the distance
def point_of_crossing_x(row):

    z_vals = [float(z) for z in row['ball_pos_z'].split('_')]
    x_vals = [float(x) for x in row['ball_pos_x'].split('_')]

    diffs_z = [abs(z - row['target_position_z']) for z in z_vals]
    idx = diffs_z.index(min(diffs_z))

    x_at_closest = x_vals[idx]

    return (x_at_closest - row['target_position_x']) * 100 # convert to cm


# check number of ppid per set order
def N_by_set_order(data):

    # check reqiured cols
    assert 'ppid' in data.columns and 'set_order' in data.columns, \
            "Data must contain 'ppid' and 'set_order'"

    ppid_list = data['ppid']

    print(f'Number of Participants in data: {len(ppid_list.unique())}')

    # check number of ppid per set order:
    counts = np.array(data.groupby('set_order')['ppid'].nunique().reset_index())
    
    # set order 1 n
    n_1 = counts[0,1]
    print(f'{n_1} ppid in set order 1')
    
    # set order 2 n
    n_2 = counts[1,1]
    print(f'{n_2} ppid in set order 2')


# grab baseline data and summarize
def summarize_phase(data, phase):

    # filter so we only have baseline data, make a copy
    phase_df = data.copy()
    phase_df = data[data['phase'] == phase]


    # filter trials given n
    #phase_df = phase_df[phase_df['trial_num_target'] >= trial_start]

    
    #print(base['ball_dist_to_center_cm'].min())
    
    #base = base.sort_values(['ppid','target_x_label','trial_num_target'])

    # summarize baseline data by ppid x target and obtain baseline errors
    phase_df_summary = (
        phase_df.groupby(['ppid','target_x_label'], as_index=False, observed=True).agg(
            mean_ball_dist_cm=('ball_dist_to_center_cm','mean'),
            min_ball_dist_cm=('ball_dist_to_center_cm','min'),
            mean_ball_dist_x_cm=('error_x_plane','mean'),
            mean_ball_launch_dev=('launch_deviation','mean'),
            mean_ball_launch_speed=('launch_Speed','mean')
        )
    )
    
    return phase_df_summary



# def add_training_status(data):

    #exp = data.query('trial_num_target > 13 and water_speed_binary == 1').copy()

   # exp.loc[exp



