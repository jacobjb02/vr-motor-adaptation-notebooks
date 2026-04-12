import numpy as np
import pandas as pd
    
import numpy as np
import pandas as pd
    
def baseline_correct(data, phase_col, baseline_string, y_col, PCA_col, 
                     min_trial=21, max_trial=52, grouping_vars=['ppid_full','target_x_label']):
    
    # 1. Filter for the STABLE baseline window only (i.e., trials 21-52)
    df_filtered = data[(data[phase_col] == baseline_string) & 
                       (data['trial_num'].between(min_trial, max_trial))]
    
    # 2. Means and Medians for BOTH columns
    df_baseline = df_filtered.groupby(grouping_vars, observed=True)[[y_col, PCA_col]].agg(['mean', 'median'])
    
    # Flatten the multi-index columns
    df_baseline.columns = [f"{col[0]}_{col[1]}" for col in df_baseline.columns]
    df_baseline = df_baseline.reset_index()
    
    # 3. Merge and Subtract
    df_merged = pd.merge(data, df_baseline, on=grouping_vars, how='left')
    
    # Subtract y_col baseline from y_col
    df_merged[f"{y_col}_mean_bc"] = df_merged[y_col] - df_merged[f"{y_col}_mean"]
    df_merged[f"{y_col}_median_bc"] = df_merged[y_col] - df_merged[f"{y_col}_median"]
    
    # Subtract PCA_col baseline from PCA_col
    df_merged[f"{PCA_col}_mean_bc"] = df_merged[PCA_col] - df_merged[f"{PCA_col}_mean"]
    df_merged[f"{PCA_col}_median_bc"] = df_merged[PCA_col] - df_merged[f"{PCA_col}_median"]
    
    # 4. BC Assertion 
    check_mask = (df_merged[phase_col] == baseline_string) & (df_merged['trial_num'].between(min_trial, max_trial))
        
    # Assert Median is zeroed for BOTH
    assert np.isclose(df_merged[check_mask].groupby(grouping_vars, observed=True)[f"{y_col}_median_bc"].median(), 0, atol=1e-8).all()
    assert np.isclose(df_merged[check_mask].groupby(grouping_vars, observed=True)[f"{PCA_col}_median_bc"].median(), 0, atol=1e-8).all()
            
    # Assert Mean is zeroed for BOTH
    assert np.isclose(df_merged[check_mask].groupby(grouping_vars, observed=True)[f"{y_col}_mean_bc"].mean(), 0, atol=1e-8).all()
    assert np.isclose(df_merged[check_mask].groupby(grouping_vars, observed=True)[f"{PCA_col}_mean_bc"].mean(), 0, atol=1e-8).all()
        
    return df_merged