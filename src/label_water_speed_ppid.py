import numpy as np
import pandas as pd



def label_water_condtion_ppid(data, 
                              ppid_col, 
                              water_col,
                              selected_speeds = np.array([])
                             ):

    data_copy = data.copy()

    # filter for requested water_speeds
    selected_speeds_mask = data_copy[water_col].isin(selected_speeds)
    data_subset = data_copy[selected_speeds_mask]
    
    
    data_subset['speed_label'] = data_subset.groupby([ppid_col,'experiment'])[water_col].transform('min')

    def encode_speed(vals):

        v = float(vals)
        if v < 0:
            return f"neg{abs(int(v))}"
        else:
            return f"pos{int(v)}"

        

    data_subset['speed_label_safe'] = data_subset['speed_label'].apply(encode_speed)

    
    # create combined ppid label:
    data_subset['ppid_full'] = (
        data_subset[ppid_col].astype(str)
        + "_" +
        data_subset['speed_label_safe'].astype(str)
    )

    return data_subset

