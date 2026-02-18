

import numpy as np
import pandas as pd



def convert_to_cm(data,
                  cols
               ):
    
    subset = data[cols].apply(pd.to_numeric, errors='coerce')
    
    # math
    converted = subset * 100
    converted = converted.add_suffix('_cm')

    # clean up and join
    data = data.drop(columns=[c for c in converted.columns if c in data.columns], errors='ignore')
    return data.join(converted)